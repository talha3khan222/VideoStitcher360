__all__ = ['writer']

from docutils import nodes

from docutils.writers.latex2e import (Writer, LaTeXTranslator,
                                      PreambleCmds)

from rstmath import mathEnv

from options import options

from code_block import CodeBlock
from docutils.parsers.rst import directives
directives.register_directive('code-block', CodeBlock)

try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict

PreambleCmds.float_settings = '''
\\usepackage{caption}
\\usepackage{float}
'''


class Translator(LaTeXTranslator):
    def __init__(self, *args, **kwargs):
        LaTeXTranslator.__init__(self, *args, **kwargs)

        # Handle author declarations

        self.current_field = ''

        self.author_names = []
        self.author_institutions = []
        self.author_emails = []
        self.paper_title = ''
        self.abstract_text = []
        self.keywords = ''
        self.table_caption = []
        self.video_url = ''

        self.abstract_in_progress = False
        self.non_breaking_paragraph = False

        self.figure_type = 'figure'
        self.figure_alignment = 'left'
        self.table_type = 'table'

    def visit_docinfo(self, node):
        pass

    def depart_docinfo(self, node):
        pass

    def visit_author(self, node):
        self.author_names.append(self.encode(node.astext()))
        raise nodes.SkipNode

    def depart_author(self, node):
        pass

    def visit_classifier(self, node):
        pass

    def depart_classifier(self, node):
        pass

    def visit_field_name(self, node):
        self.current_field = node.astext()
        raise nodes.SkipNode

    def visit_field_body(self, node):
        text = self.encode(node.astext())

        if self.current_field == 'email':
            self.author_emails.append(text)
        elif self.current_field == 'institution':
            self.author_institutions.append(text)
        elif self.current_field == 'video':
            self.video_url = text

        self.current_field = ''

        raise nodes.SkipNode

    def depart_field_body(self, node):
        raise nodes.SkipNode

    def depart_document(self, node):
        self.out.append(r'\bibliography{refs}')

        LaTeXTranslator.depart_document(self, node)

        ## Generate footmarks

        # build map: institution -> (author1, author2)
        institution_authors = OrderedDict()
        for auth, inst in zip(self.author_names, self.author_institutions):
            institution_authors.setdefault(inst, []).append(auth)

        def footmark(n):
            """Insert footmark #n.  Footmark 1 is reserved for
            the corresponding author.
            """
            return ('\\setcounter{footnotecounter}{%d}' % n,
                    '%d' % n)

        # Build one footmark for each institution
        institute_footmark = {}
        for i, inst in enumerate(institution_authors):
            institute_footmark[inst] = footmark(i + 2)

        corresponding_auth_template = r'''%%
          \affil[%(footmark_counter)s]{%
          Corresponding author: \protect\href{mailto:%(email)s}{%(email)s}}'''

        title = self.paper_title
        authors = []
        institutions_mentioned = set()
        for n, (auth, inst) in enumerate(zip(self.author_names,
                                             self.author_institutions)):
            # Corresponding author
            if n == 0:
                authors += [r'\author[%d,%d]{%s}' % (n, n + 1, auth)]
                authors += [r'\affil[0]{%s}' % self.author_emails[0]]
            else:
                authors += [r'\author[%d]{%s}' % (n + 1, auth)]

            authors += [r'\affil[%d]{%s}' % (n + 1, inst)]

        title_template = r'\newcounter{footnotecounter}' \
                r'\title{%s}%s\maketitle'
        title_template = title_template % (title, '\n'.join(authors))

        self.body_pre_docinfo = ['\n'.join(self.abstract_text) + title_template]

        # Save paper stats
        self.document.stats = {'title': title,
                               'authors': ', '.join(self.author_names),
                               'author': self.author_names,
                               'author_email': self.author_emails,
                               'author_institution': self.author_institutions,
                               'abstract': self.abstract_text,
                               'keywords': self.keywords}

    def end_open_abstract(self, node):
        if 'abstract' not in node['classes'] and self.abstract_in_progress:
            self.abstract_text.append('\\end{abstract}')
            self.abstract_in_progress = False

        elif self.abstract_in_progress:
            self.abstract_text.append(self.encode(node.astext()))

    def visit_title(self, node):
        self.end_open_abstract(node)

        if self.section_level == 1:
            if self.paper_title:
                import warnings
                warnings.warn(RuntimeWarning("Title set twice--ignored. "
                                             "Could be due to ReST"
                                             "error.)"))
            else:
                self.paper_title = self.encode(node.astext())
            raise nodes.SkipNode

        elif node.astext() == 'References':
            raise nodes.SkipNode

        LaTeXTranslator.visit_title(self, node)

    def visit_paragraph(self, node):
        self.end_open_abstract(node)

        if 'abstract' in node['classes'] and not self.abstract_in_progress:
            self.abstract_text.append('\\begin{abstract}')
            self.abstract_text.append(self.encode(node.astext()))
            self.abstract_in_progress = True

        elif 'keywords' in node['classes']:
            self.latex_preamble.append(r'\keywords{%s}' % node.astext())
            self.keywords = self.encode(node.astext())

        elif self.non_breaking_paragraph:
            self.non_breaking_paragraph = False

        else:
            self.out.append('\n\n')

    def depart_paragraph(self, node):
        pass
        ## if 'keywords' in node['classes']:
        ##     self.out.append('}')

    def visit_figure(self, node):
        self.requirements['float_settings'] = PreambleCmds.float_settings

        self.figure_type = 'figure'
        if 'classes' in node.attributes:
            placements = '[%s]' % ''.join(node.attributes['classes'])
            if 'w' in placements:
                placements = placements.replace('w', '')
                self.figure_type = 'figure*'

        self.out.append('\\begin{%s}%s' % (self.figure_type, placements))

        if node.get('ids'):
            self.out += ['\n'] + self.ids_to_labels(node)

        self.figure_alignment = node.attributes.get('align', 'center')

    def depart_figure(self, node):
        self.out.append('\\end{%s}' % self.figure_type)

    def visit_image(self, node):
        align = self.figure_alignment or 'center'
        scale = node.attributes.get('scale', None)
        filename = node.attributes['uri']

        if self.figure_type == 'figure*':
            width = r'\textwidth'
        else:
            width = r'\columnwidth'

        figure_opts = []

        if scale is not None:
            figure_opts.append('scale=%.2f' % (scale / 100.))

        # Only add \columnwidth if scale or width have not been specified.
        if 'scale' not in node.attributes and 'width' not in node.attributes:
            figure_opts.append(r'width=\columnwidth')

        self.out.append(r'\noindent\makebox[%s][%s]' % (width, align[0]))
        self.out.append(r'{\includegraphics[%s]{%s}}' % (','.join(figure_opts),
                                                         filename))

    def visit_footnote(self, node):
        # Work-around for a bug in docutils where
        # "%" is prepended to footnote text
        LaTeXTranslator.visit_footnote(self, node)
        self.out[-1] = self.out[1].strip('%')

        self.non_breaking_paragraph = True

    def visit_table(self, node):
        classes = node.attributes.get('classes', [])
        if 'w' in classes:
            self.table_type = 'table*'
        else:
            self.table_type = 'table'

        self.out.append(r'\begin{%s}' % self.table_type)
        LaTeXTranslator.visit_table(self, node)

    def depart_table(self, node):
        LaTeXTranslator.depart_table(self, node)

        self.out.append(r'\caption{%s}' % ''.join(self.table_caption))
        self.table_caption = []

        self.out.append(r'\end{%s}' % self.table_type)
        self.active_table.set('preamble written', 1)

    def visit_thead(self, node):
        # Store table caption locally and then remove it
        # from the table so that docutils doesn't render it
        # (in the wrong place)
        self.table_caption = self.active_table.caption
        self.active_table.caption = []

        opening = self.active_table.get_opening()
        opening = opening.replace('linewidth', 'tablewidth')
        self.active_table.get_opening = lambda: opening

        LaTeXTranslator.visit_thead(self, node)

    def depart_thead(self, node):
        LaTeXTranslator.depart_thead(self, node)

    def visit_literal_block(self, node):
        self.non_breaking_paragraph = True

        if 'language' in node.attributes:
            # do highlighting
            from pygments import highlight
            from pygments.lexers import get_lexer_by_name
            from pygments.formatters import LatexFormatter

            extra_opts = 'fontsize=\\footnotesize'

            linenos = node.attributes.get('linenos', False)
            linenostart = node.attributes.get('linenostart', 1)
            if linenos:
                extra_opts += ',xleftmargin=2.25mm,numbersep=3pt'

            lexer = get_lexer_by_name(node.attributes['language'])
            tex = highlight(node.astext(), lexer,
                            LatexFormatter(linenos=linenos,
                                           linenostart=linenostart,
                                           verboptions=extra_opts))

            self.out.append(tex)
            raise nodes.SkipNode
        else:
            LaTeXTranslator.visit_literal_block(self, node)

    def depart_literal_block(self, node):
        LaTeXTranslator.depart_literal_block(self, node)

    def visit_block_quote(self, node):
        self.out.append('\\begin{quotation}')
        LaTeXTranslator.visit_block_quote(self, node)

    def depart_block_quote(self, node):
        LaTeXTranslator.depart_block_quote(self, node)
        self.out.append('\\end{quotation}')

    # Math directives from rstex

    def visit_InlineMath(self, node):
        self.requirements['amsmath'] = r'\usepackage{amsmath}'
        self.out.append('$' + node['latex'] + '$')
        raise nodes.SkipNode

    def visit_PartMath(self, node):
        self.requirements['amsmath'] = r'\usepackage{amsmath}'
        self.out.append(mathEnv(node['latex'], node['label'], node['type']))
        self.non_breaking_paragraph = True
        raise nodes.SkipNode

    def visit_PartLaTeX(self, node):
        if node["usepackage"]:
            for package in node["usepackage"]:
                self.requirements[package] = r'\usepackage{%s}' % package
        self.out.append("\n" + node['latex'] + "\n")
        raise nodes.SkipNode

    def visit_citation(self, node):
        # Using BiBTeX
        raise nodes.SkipNode

    def visit_citation_reference(self, node):
        self.out.append(r"\citep{%s}" % node.astext())
        raise nodes.SkipNode

    def depart_citation_reference(self, node):
        pass



writer = Writer()
writer.translator_class = Translator
