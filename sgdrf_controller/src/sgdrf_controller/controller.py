from enum import Enum
from transitions import Machine, State, Transition
from typing import Callable
import rospy


class Controller:
    def __init__(self) -> None:
        states = [
            State(name="off", on_enter=self.off_on_enter, on_exit=self.off_on_exit),
            State(
                name="error", on_enter=self.error_on_enter, on_exit=self.error_on_exit
            ),
            State(name="idle", on_enter=self.idle_on_enter, on_exit=self.idle_on_exit),
            State(
                name="sample_standard",
                on_enter=self.sample_standard_on_enter,
                on_exit=self.sample_standard_on_exit,
            ),
            State(
                name="debubble_standard",
                on_enter=self.debubble_standard_on_enter,
                on_exit=self.debubble_standard_on_exit,
            ),
            State(
                name="sample_adaptive",
                on_enter=self.sample_adaptive_on_enter,
                on_exit=self.sample_adaptive_on_exit,
            ),
            State(
                name="debubble_adaptive",
                on_enter=self.debubble_adaptive_on_enter,
                on_exit=self.debubble_adaptive_on_exit,
            ),
            State(
                name="at_top",
                on_enter=self.at_top_on_enter,
                on_exit=self.at_top_on_exit,
            ),
            State(
                name="turning_to_go_south",
                on_enter=self.turning_to_go_south_on_enter,
                on_exit=self.turning_to_go_south_on_exit,
            ),
            State(
                name="ready_to_go_south",
                on_enter=self.ready_to_go_south_on_enter,
                on_exit=self.ready_to_go_south_on_exit,
            ),
            State(
                name="at_bottom",
                on_enter=self.at_bottom_on_enter,
                on_exit=self.at_bottom_on_exit,
            ),
            State(
                name="turning_to_go_north",
                on_enter=self.turning_to_go_north_on_enter,
                on_exit=self.turning_to_go_north_on_exit,
            ),
            State(
                name="ready_to_go_north",
                on_enter=self.ready_to_go_north_on_enter,
                on_exit=self.ready_to_go_north_on_exit,
            ),
        ]
        transitions = [
            {
                "trigger": "begin_debubble_adaptive",
                "source": ["sample_adaptive", "ready_to_go_south"],
                "dest": "debubble_adaptive",
                "prepare": None,
                "conditions": None,
                "before": None,
                "after": None,
            },
            {
                "trigger": "begin_sample_adaptive",
                "source": "debubble_adaptive",
                "dest": "sample_adaptive",
                "prepare": None,
                "conditions": None,
                "before": None,
                "after": None,
            },
            {
                "trigger": "finish_going_south",
                "source": ["sample_adaptive", "debubble_adaptive"],
                "dest": "at_bottom",
                "prepare": None,
                "conditions": None,
                "before": None,
                "after": None,
            },
            {
                "trigger": "turn_to_go_north",
                "source": "at_bottom",
                "dest": "turning_to_go_north",
                "prepare": None,
                "conditions": None,
                "before": None,
                "after": None,
            },
            {
                "trigger": "finish_turning_to_go_north",
                "source": "turning_to_go_north",
                "dest": "ready_to_go_north",
                "prepare": None,
                "conditions": None,
                "before": None,
                "after": None,
            },
            {
                "trigger": "begin_debubble_standard",
                "source": ["sample_standard", "ready_to_go_north"],
                "dest": "debubble_standard",
                "prepare": None,
                "conditions": None,
                "before": None,
                "after": None,
            },
            {
                "trigger": "begin_sample_standard",
                "source": "debubble_standard",
                "dest": "sample_standard",
                "prepare": None,
                "conditions": None,
                "before": None,
                "after": None,
            },
            {
                "trigger": "finish_going_north",
                "source": ["sample_standard", "debubble_standard"],
                "dest": "at_top",
                "prepare": None,
                "conditions": None,
                "before": None,
                "after": None,
            },
            {
                "trigger": "turn_to_go_south",
                "source": "at_top",
                "dest": "turning_to_go_south",
                "prepare": None,
                "conditions": None,
                "before": None,
                "after": None,
            },
            {
                "trigger": "finish_turning_to_go_south",
                "source": "turning_to_go_south",
                "dest": "ready_to_go_south",
                "prepare": None,
                "conditions": None,
                "before": None,
                "after": None,
            },
            {
                "trigger": "to_error",
                "source": "*",
                "dest": "error",
                "prepare": None,
                "conditions": None,
                "before": None,
                "after": None,
            },
            {
                "trigger": "begin_idling",
                "source": "*",
                "dest": "idle",
                "prepare": None,
                "conditions": None,
                "before": None,
                "after": None,
            },
            {
                "trigger": "turn_off",
                "source": "*",
                "dest": "off",
                "prepare": None,
                "conditions": None,
                "before": None,
                "after": None,
            },
            {
                "trigger": "turn_on",
                "source": "off",
                "dest": "idle",
                "prepare": None,
                "conditions": None,
                "before": None,
                "after": None,
            },
            {
                "trigger": "idle_to_debubble_adaptive",
                "source": "idle",
                "dest": "debubble_adaptive",
                "prepare": None,
                "conditions": None,
                "before": None,
                "after": None,
            },
            {
                "trigger": "idle_to_debubble_adaptive",
                "source": "idle",
                "dest": "debubble_standard",
                "prepare": None,
                "conditions": None,
                "before": None,
                "after": None,
            },
            # {'trigger': '', 'source': '', 'dest': '', 'prepare': None, 'conditions': None, 'before': None, 'after': None},
        ]
        self.machine = Machine(
            model=self,
            states=states,
            initial="off",
            prepare_event=self.prepare_event,
            before_state_change=self.before_state_change,
            after_state_change=self.after_state_change,
            finalize_event=self.finalize_event,
            auto_transitions=False,
        )
        self.machine.add_transitions(transitions=transitions)

    def prepare_event(self, *args, **kwargs):
        pass

    def before_state_change(self, *args, **kwargs):
        pass

    def after_state_change(self, *args, **kwargs):
        pass

    def finalize_event(self, *args, **kwargs):
        pass

    def off_on_enter(self, *args, **kwargs):
        pass

    def off_on_exit(self, *args, **kwargs):
        pass

    def error_on_enter(self, *args, **kwargs):
        pass

    def error_on_exit(self, *args, **kwargs):
        pass

    def idle_on_enter(self, *args, **kwargs):
        pass

    def idle_on_exit(self, *args, **kwargs):
        pass

    def sample_standard_on_enter(self, *args, **kwargs):
        pass

    def sample_standard_on_exit(self, *args, **kwargs):
        pass

    def debubble_standard_on_enter(self, *args, **kwargs):
        pass

    def debubble_standard_on_exit(self, *args, **kwargs):
        pass

    def sample_adaptive_on_enter(self, *args, **kwargs):
        pass

    def sample_adaptive_on_exit(self, *args, **kwargs):
        pass

    def debubble_adaptive_on_enter(self, *args, **kwargs):
        pass

    def debubble_adaptive_on_exit(self, *args, **kwargs):
        pass

    def at_top_on_enter(self, *args, **kwargs):
        pass

    def at_top_on_exit(self, *args, **kwargs):
        pass

    def turning_to_go_south_on_enter(self, *args, **kwargs):
        pass

    def turning_to_go_south_on_exit(self, *args, **kwargs):
        pass

    def ready_to_go_south_on_enter(self, *args, **kwargs):
        pass

    def ready_to_go_south_on_exit(self, *args, **kwargs):
        pass

    def at_bottom_on_enter(self, *args, **kwargs):
        pass

    def at_bottom_on_exit(self, *args, **kwargs):
        pass

    def turning_to_go_north_on_enter(self, *args, **kwargs):
        pass

    def turning_to_go_north_on_exit(self, *args, **kwargs):
        pass

    def ready_to_go_north_on_enter(self, *args, **kwargs):
        pass

    def ready_to_go_north_on_exit(self, *args, **kwargs):
        pass
