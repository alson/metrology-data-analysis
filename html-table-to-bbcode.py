#!/usr/bin/env python

import re
import sys


html_contents = sys.stdin.read()

html_contents = re.sub(r'^.*?<table', r'<table', html_contents, flags=re.DOTALL)
html_contents = re.sub(r'^(.*)</table>.*?$', r'\1</table>', html_contents, flags=re.DOTALL)
html_contents = re.sub(r'</table>.+?<table', r'</table>\n<table', html_contents, flags=re.DOTALL)
html_contents = re.sub(r'<table[^>]*>', r'[table]', html_contents)
html_contents = re.sub(r'</table[^>]*>', r'[/table]', html_contents)
html_contents = re.sub(r'^\s*</?thead>$', r'', html_contents, flags=re.MULTILINE)
html_contents = re.sub(r'^\s*</?tbody>$', r'', html_contents, flags=re.MULTILINE)
html_contents = re.sub(r'<th[^>]*>', r'[td][b]', html_contents)
html_contents = re.sub(r'</th[^>]*>', r'[/b][/td]', html_contents)
html_contents = re.sub(r'<tr[^>]*>', r'[tr]', html_contents)
html_contents = re.sub(r'</tr[^>]*>', r'[/tr]', html_contents)
html_contents = re.sub(r'<td[^>]*>', r'[td]', html_contents)
html_contents = re.sub(r'</td[^>]*>', r'[/td]', html_contents)

print(html_contents)
