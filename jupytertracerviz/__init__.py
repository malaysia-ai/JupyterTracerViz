from IPython.display import IFrame, display, HTML
from html import escape
from viztracer import report_builder
from string import Template
import viztracer
import os
import json
from .template import base
from .multigpus_repl import multigpus, init_multigpus_repl

def visualize(files, width = '100%', height = '1024'):
    if isinstance(files, str):
        files = [files]
    data = []
    for f in files:
        data.append(report_builder.get_json(f))
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