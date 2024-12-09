from custom_interfaces.msg import SubsystemState

class SubsystemPoller:

    def __init__(self, node, key: str):
        self.node = node
        self.node.enabled = False
        self.key = key
        self.node.create_subscription(SubsystemState, "biomimic/subsystem_state", self.callback, 10)

    def callback(self, msg: SubsystemState):
        val = getattr(msg, self.key, None)
        if val is None:
            self.node.get_logger().fatal(f"Key: {self.key} does not exist in subsystem state poller", skip_first=False, throttle_duration_sec=2.0)
            raise Exception("Unsupported subsystem state key")
        self.node.enabled = val

