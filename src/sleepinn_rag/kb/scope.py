"""Structural context used by the SCOPE retrieval variants."""
import os
import re
from sleepinn_rag import config

def infer_chunk_role(chunk_text: str, metadata: dict=None) -> str:
    if not config.USE_CHUNK_ROLE_TAGS:
        return 'paragraph'
    text = chunk_text.strip()
    lines = text.split('\n')
    num_lines = len(lines)
    if num_lines == 0:
        return 'paragraph'
    dotted = sum((1 for l in lines if '..' in l or re.search('\\d+\\s*$', l.strip())))
    if num_lines >= 3 and dotted / num_lines > 0.5:
        return 'toc'
    if num_lines <= 2 and len(text) < 120:
        stripped = text.strip().rstrip(':')
        if stripped.isupper() or stripped.istitle():
            return 'heading'
    pipe_lines = sum((1 for l in lines if '|' in l))
    tab_lines = sum((1 for l in lines if '\t' in l))
    if num_lines >= 3 and (pipe_lines / num_lines > 0.4 or tab_lines / num_lines > 0.4):
        return 'table'
    lower = text.lower()
    if lower.startswith('figure') or lower.startswith('fig.') or lower.startswith('fig '):
        return 'figure'
    return 'paragraph'

def extract_document_structure(pages: list) -> dict:
    structure = {}
    current = {'doc_title': '', 'chapter': '', 'section': '', 'subsection': ''}
    prev_source = None
    for page in pages:
        meta = getattr(page, 'metadata', {}) or {}
        source = os.path.basename(meta.get('source', 'unknown'))
        page_num = meta.get('page', None)
        text = page.page_content or ''
        if source != prev_source:
            current = {'doc_title': os.path.splitext(source)[0], 'chapter': '', 'section': '', 'subsection': ''}
            prev_source = source
        first_lines = text.split('\n')[:8]
        for line in first_lines:
            line = line.strip()
            if not line or len(line) < 3:
                continue
            if re.match('^(?:chapter|part)\\s+[\\divxlc]+', line, re.IGNORECASE):
                current['chapter'] = line
                current['section'] = ''
                current['subsection'] = ''
                break
            if re.match('^\\d+\\.\\d+\\s+[A-Z]', line):
                current['section'] = line
                current['subsection'] = ''
                break
            if re.match('^\\d+\\.\\s+[A-Z]', line):
                if not current['chapter']:
                    current['chapter'] = line
                else:
                    current['section'] = line
                current['subsection'] = ''
                break
            if current['section'] and line.istitle() and (len(line) < 80) and (not line.endswith('.')):
                current['subsection'] = line
                break
            if line.isupper() and 5 < len(line) < 100:
                if not current['chapter']:
                    current['chapter'] = line
                elif not current['section']:
                    current['section'] = line
                else:
                    current['subsection'] = line
                break
        structure[source, page_num] = dict(current)
    return structure

def build_contextualized_chunk(chunk_text, doc_title='', chapter='', section='', subsection='', page=None, chunk_type='paragraph', style=None) -> str:
    if not config.USE_CONTEXTUAL_CHUNKS:
        return chunk_text
    style = style or config.CONTEXTUAL_CHUNK_STYLE
    if style == 'compact':
        parts = []
        if config.INCLUDE_DOC_TITLE and doc_title:
            parts.append(doc_title)
        if config.INCLUDE_CHAPTER and chapter:
            parts.append(chapter)
        if config.INCLUDE_SECTION and section:
            parts.append(section)
        if config.INCLUDE_SUBSECTION and subsection:
            parts.append(subsection)
        suffix = []
        if config.INCLUDE_PAGE and page is not None:
            suffix.append(f'Page {page}')
        if config.INCLUDE_CHUNK_TYPE and chunk_type and (chunk_type != 'paragraph'):
            suffix.append(chunk_type)
        header = ' > '.join(parts)
        if suffix:
            header += ' | ' + ' | '.join(suffix)
        return f'{header}\n\n{chunk_text}' if header else chunk_text
    else:
        lines = []
        if config.INCLUDE_DOC_TITLE and doc_title:
            lines.append(f'Document: {doc_title}')
        if config.INCLUDE_CHAPTER and chapter:
            lines.append(f'Chapter: {chapter}')
        if config.INCLUDE_SECTION and section:
            lines.append(f'Section: {section}')
        if config.INCLUDE_SUBSECTION and subsection:
            lines.append(f'Subsection: {subsection}')
        if config.INCLUDE_PAGE and page is not None:
            lines.append(f'Page: {page}')
        if config.INCLUDE_CHUNK_TYPE and chunk_type and (chunk_type != 'paragraph'):
            lines.append(f'Chunk type: {chunk_type}')
        if lines:
            return '\n'.join(lines) + '\n\n' + chunk_text
        return chunk_text
