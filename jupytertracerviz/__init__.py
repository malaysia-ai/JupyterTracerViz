from IPython.display import IFrame, display, HTML
from html import escape
from viztracer import report_builder
from string import Template
import viztracer
import os
import json
from .template import base
from .multigpus_repl import multigpus, init_multigpus_repl
from typing import Union, List

def visualize(
    files: Union[str, List[str]], 
    width: str = '100%', 
    height: str = '1024',
    remove_spans: List[str] = [
        'ipykernel', 
        'IPython', 
        'jupytertracerviz', 
        'multiprocessing', 
        'asyncio', 
        'tornado',
        'torch/profiler',
        'traitlets/',
        'runpy.py',
        '<string>',
        'Unrecognized',
        'threading.py',
    ],
    merge_overlap: bool = True,
):
    """
    Visualizes one or more VizTracer trace files in an embedded HTML iframe within a Jupyter Notebook.

    Parameters:
    ----------
    files : Union[str, List[str]]
        Path(s) to one or more `.json` VizTracer trace files to visualize. Can be a single file path or a list of file paths.

    width : str, optional
        Width of the embedded viewer (default: "100%"). Accepts any valid CSS width (e.g., "800px", "100%").

    height : str, optional
        Height of the embedded viewer in pixels (default: "1024").

    Returns:
    -------
    IPython.display.HTML
        An HTML iframe displaying the interactive VizTracer trace viewer within the notebook.

    Notes:
    -----
    - This function uses VizTracer's built-in HTML viewer templates to embed a rich profiling UI directly in the notebook.
    - Automatically handles escaping of embedded script tags to avoid HTML rendering issues.
    - Combines multiple trace files into a single view if a list is provided.
    - Requires VizTracer and IPython to be installed.
    """

    if isinstance(files, str):
        files = [files]
    data = []
    exists = set()
    for f in files:
        d = report_builder.get_json(f)
        for i in reversed(range(len(d['traceEvents']))):
            if any([s in d['traceEvents'][i]['name'] for s in remove_spans]):
                d['traceEvents'].pop(i)
            key = d['traceEvents'][i]['name'] + str(d['traceEvents'][i]['pid']) + str(i)
            if key in exists and merge_overlap:
                d['traceEvents'].pop(i)
                continue
            exists.add(key)
        data.append(d)
    builder = report_builder.ReportBuilder(data)
    builder.prepare_json(file_info=True, display_time_unit="ns")
    sub = {}
    with open(os.path.join(os.path.dirname(viztracer.__file__), "html/trace_viewer_embedder.html"), encoding="utf-8") as f:
        tmpl = f.read()
    with open(os.path.join(os.path.dirname(viztracer.__file__), "html/trace_viewer_full.html"), encoding="utf-8") as f:
        sub["trace_viewer_full"] = f.read()
    sub["json_data"] = json.dumps(builder.combined_json).replace("</script>", "<\\/script>")
    html = Template(base).substitute(sub)
    iframe = (
        '<div style="width:{width};height:{height}px">'
        '<div style="position:relative;width:100%;height:0;padding-bottom:{ratio};">'  # noqa
        '<iframe srcdoc="{html}" style="position:absolute;width:100%;height:{height}px;left:0;top:0;'  # noqa
        'border:none !important;" '
        'allowfullscreen webkitallowfullscreen mozallowfullscreen>'
        '</iframe>'
        '</div></div>'
    ).format(html=escape(html), width=width, height=height, ratio=1)
    return display(HTML(iframe))