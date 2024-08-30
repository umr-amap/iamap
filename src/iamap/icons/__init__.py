import os
from PyQt5.QtGui import QIcon

cwd = os.path.abspath(os.path.dirname(__file__))
# encoder_tool_path = os.path.join(cwd, 'encoder_tool.svg')
encoder_tool_path = os.path.join(cwd, 'network-base-solid-svgrepo-com.svg')
reduction_tool_path = os.path.join(cwd, 'chart-scatterplot-solid-svgrepo-com.svg')
cluster_tool_path = os.path.join(cwd, 'chart-scatter-3d-svgrepo-com.svg')
similarity_tool_path = os.path.join(cwd, 'scatter-plot-svgrepo-com.svg')
random_forest_tool_path = os.path.join(cwd, 'forest-svgrepo-com.svg')

QIcon_EncoderTool = QIcon(encoder_tool_path)
QIcon_ReductionTool = QIcon(reduction_tool_path)
QIcon_ClusterTool = QIcon(cluster_tool_path)
QIcon_SimilarityTool = QIcon(similarity_tool_path)
QIcon_RandomforestTool = QIcon(random_forest_tool_path)
