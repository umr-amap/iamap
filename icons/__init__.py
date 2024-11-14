import os
from PyQt5.QtGui import QIcon

cwd = os.path.abspath(os.path.dirname(__file__))
encoder_tool_path = os.path.join(cwd, "encoder.svg")
reduction_tool_path = os.path.join(cwd, "proj.svg")
cluster_tool_path = os.path.join(cwd, "cluster.svg")
similarity_tool_path = os.path.join(cwd, "sim.svg")
random_forest_tool_path = os.path.join(cwd, "forest.svg")

QIcon_EncoderTool = QIcon(encoder_tool_path)
QIcon_ReductionTool = QIcon(reduction_tool_path)
QIcon_ClusterTool = QIcon(cluster_tool_path)
QIcon_SimilarityTool = QIcon(similarity_tool_path)
QIcon_RandomforestTool = QIcon(random_forest_tool_path)
