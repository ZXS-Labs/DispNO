from models.gwcnet_pianchaHorizonDef import GwcNet_G, GwcNet_GC
from models.gwcnet_pianchaHorizonDefDispNO import GwcNet_G_DispNO, GwcNet_GC_DispNO
from models.gwcnet_pianchaHorizonDefDispNORange import GwcNet_G_DispNO_range, GwcNet_GC_DispNO_range
from models.loss import model_loss, unimodal_loss, bimodal_loss

__models__ = {
    "gwcnet-g": GwcNet_G,
    "gwcnet-gc": GwcNet_GC,
    "gwcnet-g-dispNO": GwcNet_G_DispNO,
    "gwcnet-gc-dispNO": GwcNet_GC_DispNO,
    "gwcnet-g-dispNO-range": GwcNet_G_DispNO_range,
    "gwcnet-gc-dispNO-range": GwcNet_GC_DispNO_range,
}
