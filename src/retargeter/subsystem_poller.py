from retargeter.retargeter_node import RetargeterNode
from custom_interfaces.msg import SubsystemState


class SubsystemPoller:

    def __init__(self, node: RetargeterNode):
        self.node: RetargeterNode = node
        self.node.create_subscription(SubsystemState, "biomimic/subsystem_state", self.callback, 10)

    def callback(self, msg: SubsystemState):
        self.node.enabled = msg.retargeter_enabled