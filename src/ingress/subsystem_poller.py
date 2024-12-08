from ingress.rokoko.rokoko_node import RokokoNode
from custom_interfaces.msg import SubsystemState


class SubsystemPoller:

    def __init__(self, node: RokokoNode):
        self.node: RokokoNode = node
        self.node.create_subscription(SubsystemState, "biomimic/subsystem_state", self.callback, 10)

    def callback(self, msg: SubsystemState):
        self.node.enabled = msg.rokoko_enabled