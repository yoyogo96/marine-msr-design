"""
MSR Design Report - Table Generation Utilities
Generates formatted parameter tables for the design report.
"""


def format_value(value, unit=''):
    """Format a numerical value with appropriate precision.

    Selects the number of significant figures based on magnitude:
      >= 1e6   -> scientific notation with 3 significant figures
      >= 100   -> 1 decimal place
      >= 1     -> 3 decimal places
      >= 0.001 -> 4 decimal places
      < 0.001  -> scientific notation with 3 significant figures

    Args:
        value: Numerical value or string to format
        unit: Optional unit string appended after the value

    Returns:
        str: Formatted value string (unit appended if provided)
    """
    if isinstance(value, str):
        return value
    if abs(value) >= 1e6:
        return f"{value:.3e} {unit}".strip()
    elif abs(value) >= 100:
        return f"{value:.1f} {unit}".strip()
    elif abs(value) >= 1:
        return f"{value:.3f} {unit}".strip()
    elif abs(value) >= 0.001:
        return f"{value:.4f} {unit}".strip()
    else:
        return f"{value:.3e} {unit}".strip()


def markdown_table(title, rows, headers=None):
    """Generate a markdown table string.

    Args:
        title: Table title/caption (rendered as ### heading)
        rows: List of row sequences; each row should contain values
              matching the number of headers. Float values are
              auto-formatted; all others are converted via str().
        headers: Column header list (default: Parameter | Value | Unit | Description)

    Returns:
        str: Complete markdown table including title heading
    """
    if headers is None:
        headers = ['Parameter', 'Value', 'Unit', 'Description']

    lines = [f"\n### {title}\n"]

    # Header row
    lines.append('| ' + ' | '.join(headers) + ' |')
    lines.append('|' + '|'.join(['---'] * len(headers)) + '|')

    # Data rows
    for row in rows:
        formatted = []
        for item in row:
            if isinstance(item, float):
                formatted.append(format_value(item))
            else:
                formatted.append(str(item))
        # Pad short rows to match header count
        while len(formatted) < len(headers):
            formatted.append('')
        lines.append('| ' + ' | '.join(formatted) + ' |')

    return '\n'.join(lines)


def parameter_summary(title, params_dict):
    """Generate a parameter summary markdown table from a dictionary.

    Args:
        title: Table title string
        params_dict: Dict mapping parameter name -> (value, unit, description)
                     e.g. {'Thermal power': (40e6, 'W', 'Reactor rated power')}

    Returns:
        str: Markdown table string
    """
    rows = []
    for name, (value, unit, desc) in params_dict.items():
        rows.append([name, format_value(value, ''), unit, desc])
    return markdown_table(title, rows)


def print_section_header(title, char='='):
    """Print a formatted section header to the console.

    Args:
        title: Section title string
        char: Border character (default '=')
    """
    width = max(60, len(title) + 4)
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}")


def print_param_table(title, params):
    """Print a formatted parameter table to the console.

    Args:
        title: Table title string
        params: List of (name, value, unit) tuples
    """
    print_section_header(title, '-')
    max_name = max(len(p[0]) for p in params) + 2
    max_val = max(len(format_value(p[1])) for p in params) + 2
    for name, value, unit in params:
        print(f"  {name:<{max_name}} {format_value(value):>{max_val}}  {unit}")
    print()


def results_to_markdown(results_dict, filename):
    """Save a nested results dictionary to a markdown file.

    The top-level keys become ## section headings. Values may be:
      - dict  -> items rendered as bullet list; values may be
                 (number, unit) tuples or plain values
      - str   -> rendered verbatim as a paragraph

    Args:
        results_dict: Dict mapping section title -> params (dict or str)
        filename: Output file path (absolute or relative to cwd)
    """
    lines = ["# MSR Design Analysis Results\n"]

    for section, params in results_dict.items():
        lines.append(f"\n## {section}\n")
        if isinstance(params, dict):
            for key, val in params.items():
                if isinstance(val, tuple):
                    value, unit = val
                    lines.append(f"- **{key}**: {format_value(value)} {unit}")
                else:
                    lines.append(f"- **{key}**: {val}")
        elif isinstance(params, str):
            lines.append(params)

    with open(filename, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"  Results saved: {filename}")
