#!/usr/bin/env python3

import os
import re
import markdown
import glob
import html

from sanitize_html import clean   # same folder: the allowlist every rendered fragment goes through


def parse_databricks_notebook(filepath):
    """Parse a Databricks .py notebook format into cells"""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Split by COMMAND ----------
    sections = re.split(r'# COMMAND ----------', content)
    cells = []
    
    for section in sections:
        if not section.strip():
            continue
            
        # Check if this is a markdown cell
        if '# MAGIC %md' in section:
            # Extract markdown content
            lines = section.split('\n')
            md_lines = []
            for line in lines:
                if line.startswith('# MAGIC %md'):
                    # Remove '# MAGIC %md'
                    md_lines.append(line[11:].strip())
                elif line.startswith('# MAGIC '):
                    # Remove '# MAGIC '
                    md_lines.append(line[8:])
                elif line.startswith('# MAGIC'):
                    # Remove '# MAGIC'
                    md_lines.append(line[7:])
            
            md_content = '\n'.join(md_lines)
            cells.append({'type': 'markdown', 'content': md_content})
        else:
            # This is a code cell
            # Remove any leading comments that aren't actual code
            lines = section.split('\n')
            code_lines = []
            for line in lines:
                if not line.startswith('# DBTITLE'):
                    code_lines.append(line)
            
            code_content = '\n'.join(code_lines).strip()
            if code_content:
                cells.append({'type': 'code', 'content': code_content})
    
    return cells


def convert_to_html_fragment(filepath):
    """Convert Databricks .py notebook to HTML fragment with syntax highlighting"""
    filename = os.path.basename(filepath)
    name_without_ext = os.path.splitext(filename)[0]
    
    cells = parse_databricks_notebook(filepath)
    html_content = []
    
    for i, cell in enumerate(cells):
        if cell['type'] == 'markdown':
            # Convert markdown to HTML using nbconvert structure
            # Sanitized: raw HTML in a Markdown cell cannot add script to the page
            md_html = clean(markdown.markdown(
                cell['content'],
                extensions=['fenced_code', 'tables', 'nl2br', 'toc']
            ))
            html_content.append(f'''<div class="cell border-box-sizing text_cell rendered">
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
{md_html}
</div>
</div>
</div>''')
        elif cell['type'] == 'code':
            # Create code cell with proper syntax highlighting for Python
            escaped_code = html.escape(cell['content'])
            html_content.append(f'''<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="inner_cell">
<div class="input_area">
<div class="highlight hl-ipython3">
<pre class="language-python"><code class="language-python">{escaped_code}</code></pre>
</div>
</div>
</div>
</div>
</div>''')
    
    # Return just the content fragment (no full HTML document)
    fragment_content = '\n'.join(html_content)
    
    # Write fragment to temp file for the main script to read
    temp_path = f"temp_{name_without_ext}_fragment.html"
    with open(temp_path, 'w') as f:
        f.write(fragment_content)
    
    return name_without_ext, fragment_content


def is_databricks_notebook(filepath):
    """Databricks .py source notebooks start with this header line; other .py files are code."""
    with open(filepath, 'r') as f:
        return f.readline().strip() == "# Databricks notebook source"


if __name__ == "__main__":
    # Every Databricks .py notebook under projects/<project>/, named <project>__<notebook>
    notebook_data = {}
    for py_file in sorted(glob.glob('projects/*/**/*.py', recursive=True)):
        if 'node_modules' in py_file.split(os.sep) or not is_databricks_notebook(py_file):
            continue
        name, fragment = convert_to_html_fragment(py_file)
        project = py_file.split(os.sep)[1]
        notebook_data[f"{project}__{name}"] = fragment
        print(f"Converted {py_file} to HTML fragment")
    
    # Write notebook data to a JSON file for the main script
    import json
    with open('notebook_fragments.json', 'w') as f:
        json.dump(notebook_data, f)