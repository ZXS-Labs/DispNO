from .kitti_dataset import KITTIDataset
from .sceneflow_dataset import SceneFlowDatset
from .sceneflow_dataset_DispNO import SceneFlowDatsetDispNO
from .kitti_dataset_DispNO import KITTIDatasetDispNO
from .middlebury_dataset import MiddleburyDataset
from .middlebury_dataset_DispNO import MiddleburyDatsetDispNO
from .UnrealStereo_dataset import UnrealStereoDatset
from .UnrealStereo_dataset_DispNO import UnrealStereoDatsetDispNO

__datasets__ = {
    "sceneflow": SceneFlowDatset,
    "sceneflowDispNO": SceneFlowDatsetDispNO, 
    "kitti": KITTIDataset,
    "kittiDispNO": KITTIDatasetDispNO,
    "middlebury": MiddleburyDataset,
    "middleburyDispNO": MiddleburyDatsetDispNO,
    "UnrealStereo": UnrealStereoDatset,
    "UnrealStereoDispNO": UnrealStereoDatsetDispNO,
}

