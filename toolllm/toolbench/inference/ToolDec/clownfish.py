"""Simple streaming JSON (Schema) parser"""

from enum import Enum
import json
from typing import Union

from pyrsistent import PRecord, PSet, s as pset, field

class Candidates(Enum):
    NONE = 0
    ANY = 1

def reduce_candidates(*args: Union[Candidates, list[str]]):
    out = []
    for arg in args:
        if arg == Candidates.NONE:
            continue
        if arg == Candidates.ANY:
            return Candidates.ANY
        out += arg
    if len(out) == 0:
        return Candidates.NONE
    return out

class NodeParser(PRecord):
    """Base class for parsers"""
    completed: bool = field()
    can_continue: bool = field()

    @staticmethod
    def init():
        """Create a parser"""

    @staticmethod
    def step(state: Union['NodeParser', None], char: str):
        """Returns fresh state if the provide char is valid"""

    @staticmethod
    def candidates(state: Union['NodeParser', None]) -> Union[Candidates, list[str]]:
        """Returns the set of possible candidates for next token"""

def parser_for_type(node, schema):
    """Returns a perse to handle a particular type"""
    if '$ref' in node:
        name = node['$ref'].split('/')[-1]
        return parser_for_type(schema['definitions'][name], schema)

    if 'enum' in node:
        return UnionParser.init([LiteralParser.init(json.dumps(v)) for v in node['enum']], schema)

    if 'anyOf' in node:
        return UnionParser.init([parser_for_type(v, schema) for v in node['anyOf']], schema)

    if 'type' in node:
        if node['type'] == 'string':
            return StringParser.init()
        if node['type'] == 'number' or node['type'] == 'integer':
            return NumberParser.init()
        if node['type'] == 'array':
            return ListParser.init(node, schema)
        if node['type'] == 'object':
            return DictParser.init(node, schema)


class NumberParser(NodeParser):
    so_far: str = field()
    completed: bool = field()
    can_continue: bool = field()

    @staticmethod
    def init():
        return NumberParser(so_far='', completed=False, can_continue=True)
    
    @staticmethod
    def step(state: Union['NumberParser', None], char: str):
        if not state:
            return None 

        is_digit = char in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        is_period = char == '.'

        if len(char) == 0:
            if not is_digit:
                return None

        if is_period and '.' in state.so_far:
            return None

        if not is_digit and not is_period:
            return None

        return state.update({
            'so_far': state.so_far + char,
            'completed': True,
            'can_continue': True
        })

    @staticmethod
    def candidates(state: Union['NumberParser', None]):
        if '.' in state.so_far:
            return ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        return ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.']

class StringState(Enum):
    """Different states in the String Parser state machine"""
    OPENING = 1
    OPENED = 2
    ESCAPING = 3
    COMPLETE = 4

class StringParser(NodeParser):
    """Parser for fully dynamic string (that can be anything)"""

    state: StringState = field()
    completed: bool = field()
    can_continue: bool = field()

    @staticmethod
    def init():
        """Set up an empty string parser"""
        return StringParser(state=StringState.OPENING, completed=False, can_continue=True)

    @staticmethod
    def step(state: Union['StringParser', None], char: str):
        """Step through one character"""
        if not state:
            return None

        if state.state == StringState.OPENING:
            if char in [' ', '\t', '\n', '\r']:
                return state

            if char == '"':
                return state.set('state', StringState.OPENED)

            return None

        if state.state == StringState.OPENED:
            if char == '\\':
                return state.set('state', StringState.ESCAPING)
            if char == '"':
                return state.update({'state': StringState.COMPLETE, 'completed': True, 'can_continue': False})
            return state

        if state.state == StringState.ESCAPING:
            return state.set('state', StringState.OPENED)

    @staticmethod
    def candidates(state: Union['StringParser', None]):
        if not state:
            return Candidates.NONE
        if state.state == StringState.OPENING:
            return ['"']
        if state.state == StringState.OPENED:
            return Candidates.ANY
        if state.state == StringState.ESCAPING:
            return Candidates.ANY
        if state.state == StringState.COMPLETE:
            return Candidates.NONE

class ListState(Enum):
    """Different states used in the list parser state machine"""
    OPENING = 1
    MAYBE_ELEM = 2
    DEFINE_ELEM = 3
    ELEM_NEXT = 4
    COMPLETE = 5

class ListParser(PRecord):
    """Parser for lists"""

    state: ListState = field()
    completed: bool = field()
    can_continue: bool = field()
    node = field()
    schema = field()
    active_element_state: NodeParser = field()

    @staticmethod
    def init(node, schema):
        """Set up an empty list parser"""
        return ListParser(
            state=ListState.OPENING,
            completed=False,
            can_continue=True,
            node=node,
            schema=schema)

    @staticmethod
    def step(state: Union['ListParser', None], char: str):
        """Step through one character"""
        if not state:
            return None

        # If we're not delegating characters to the element, skip whitespace
        if state.state != ListState.DEFINE_ELEM:
            if char in [' ', '\t', '\n', '\r']:
                return state
            
        if state.state == ListState.OPENING:
            if char == '[':
                return state.update({
                    'state': ListState.MAYBE_ELEM,
                    'active_element_state': parser_for_type(state.node['items'], state.schema)
                })
            return None

        if state.state == ListState.MAYBE_ELEM:
            if char == ']':
                return state.update({
                    'state': ListState.COMPLETE, 
                    'completed': True,
                    'can_continue': False,
                    'active_element_state': None
                })
            
            state = state.update({
                'state': ListState.DEFINE_ELEM
            })
        
        if state.state == ListState.DEFINE_ELEM:
            item = state.active_element_state.step(state.active_element_state, char)
            if not item:
                # If this isn't part of the child item, then check if we considered it
                # as if we were in ELEM_NEXT, if it would be a valid char and return that
                if state.active_element_state.completed:
                    state = state.update({
                        'state': ListState.ELEM_NEXT, 
                        'active_element_state': None
                    })
                    return state.step(state, char)

                return None

            state = state.set('active_element_state', item)
            if item.completed and not item.can_continue:
                state = state.update({
                    'state': ListState.ELEM_NEXT, 
                    'active_element_state': None
                })

            return state
        
        if state.state == ListState.ELEM_NEXT:
            if char == ',':
                return state.update({
                    'state': ListState.DEFINE_ELEM, 
                    'active_element_state': parser_for_type(state.node['items'], state.schema)
                })
            if char == ']':
                return state.update({
                    'state': ListState.COMPLETE,
                    'completed': True,
                    'can_continue': False
                })
            return None

    @staticmethod
    def candidates(state: Union['ListParser', None]) -> Union[Candidates, list[str]]:
        if not state:
            return Candidates.NONE
        
        if state.state == ListState.OPENING:
            return ['[']

        if state.state == ListState.MAYBE_ELEM:
            item_candidates = state.active_element_state.candidates(state.active_element_state)
            if item_candidates == Candidates.ANY:
                return Candidates.ANY
            if item_candidates == Candidates.NONE:
                return [']']
            return item_candidates + [']']

        if state.state == ListState.DEFINE_ELEM:
            element_candidates = state.active_element_state.candidates(state.active_element_state)
            if state.active_element_state.completed:
                return reduce_candidates(element_candidates, [',', ']'])
            return element_candidates
        
        if state.state == ListState.ELEM_NEXT:
            return [',', ']']

        if state.state == ListState.COMPLETE:
            return Candidates.NONE

class DictState(Enum):
    """Different states used in the dict parser state machine"""
    OPENING = 1
    OPENED = 2
    PICK_PROP = 3
    PROP_COLON = 4
    PROP_DEF = 5
    PROP_NEXT = 6
    COMPLETE = 7

class DictParser(PRecord):
    """Parser for dicts/objects"""

    state: DictState = field()
    active_prop: Union[str, None] = field()
    active_prop_state: Union[NodeParser, None] = field()
    defined_props: PSet = field()
    valid_props: PSet = field()
    completed: bool = field()
    can_continue: bool = field()
    node = field()
    schema = field()

    @staticmethod
    def init(node, schema):
        """Set up an empty list parser"""
        return DictParser(
            state=DictState.OPENING, 
            active_prop=None,
            active_prop_state=None,
            defined_props=pset(),
            valid_props=pset(*node['properties'].keys()),
            completed=False,
            can_continue=True,
            node=node,
            schema=schema)

    @staticmethod
    def step(state: Union['DictParser', None], char: str):
        """Step through one character"""
        if not state:
            return None

        # Handle whitespace
        if state.state in [DictState.OPENING, DictState.OPENED, DictState.PROP_COLON, DictState.PROP_NEXT]:
            if char in [' ', '\t', '\n', '\r']:
                return state

        if state.state == DictState.OPENING:
            if char == "{":
                return state.set('state', DictState.OPENED)
            
        if state.state == DictState.OPENED and len(state.valid_props) != 0:
            if char == '"':
                return state.update({'state': DictState.PICK_PROP, 'active_prop': ''})

        if state.state == DictState.PICK_PROP:
            if char == '"':
                if state.active_prop in state.valid_props:
                    return state.update({
                        'state': DictState.PROP_COLON,
                        'active_prop_state': parser_for_type(state.node['properties'][state.active_prop], state.schema)
                    })
                return None
            next_prop = state.active_prop + char
            if any([p.startswith(next_prop) for p in state.valid_props]):
                return state.set('active_prop', next_prop)
            return None

        if state.state == DictState.PROP_COLON:
            if char == ':':
                return state.set('state', DictState.PROP_DEF)
            return None

        if state.state == DictState.PROP_DEF:
            item = state.active_prop_state.step(state.active_prop_state, char)
            if not item:
                # If this isn't part of the child item, then check if we considered it
                # as if we were in PROP_NEXT, if it would be a valid char and return that
                if state.active_prop_state.completed:
                    state = state.update({
                        'state': DictState.PROP_NEXT, 
                        'valid_props': pset(*[k for k in state.valid_props if k != state.active_prop]),
                        'defined_props': state.defined_props.add(state.active_prop),
                        'active_prop_state': None,
                        'active_prop': None
                    })
                    return state.step(state, char)

                return None
            
            if item.completed and not item.can_continue:
                return state.update({
                    'state': DictState.PROP_NEXT,
                    'valid_props': pset(*[k for k in state.valid_props if k != state.active_prop]),
                    'defined_props': state.defined_props.add(state.active_prop),
                    'active_prop_state': None,
                    'active_prop': None
                })

            return state.set('active_prop_state', item)

        if state.state == DictState.PROP_NEXT:
            if char == ',':
                if len(state.valid_props):
                    return state.set('state', DictState.OPENED)
                return None
            if char == '}':
                if all([r in state.defined_props for r in state.node['required']]):
                    return state.update({
                        'state': DictState.COMPLETE,
                        'completed': True,
                        'can_continue': False
                    })
                return None
            return None
        
        if len(state.valid_props) == 0 and state.state == DictState.OPENED and char == "}":
            print("yep")
            return state.update({
                        'state': DictState.COMPLETE,
                        'completed': True,
                        'can_continue': False
                    })

    @staticmethod
    def candidates(state: Union['DictParser', None]) -> Union[Candidates, list[str]]:
        if not state:
            return Candidates.NONE

        if state.state == DictState.OPENING:
            return ['{ ']
        if state.state == DictState.OPENED:
            return ['"' + p + '": ' for p in state.valid_props]
        if state.state == DictState.PICK_PROP:
            return [p[len(state.active_prop):] + '":' for p in state.valid_props if p.startswith(state.active_prop)]
        if state.state == DictState.PROP_COLON:
            return [': ']
        if state.state == DictState.PROP_DEF:
            prop_candidates = state.active_prop_state.candidates(state.active_prop_state)
            if state.active_prop_state.completed:
                if len(state.valid_props) > 1:
                    return reduce_candidates(prop_candidates, [', ', ' }'])
                return reduce_candidates(prop_candidates, [' }'])
            return prop_candidates

        if state.state == DictState.PROP_NEXT:
            if len(state.valid_props):
                return [', ', ' }']
            return [' }']
        if state.state == DictState.COMPLETE:
            return Candidates.NONE

class LiteralParser(NodeParser):
    goal: str = field()
    so_far: str = field()
    completed: bool = field()
    can_continue: bool = field()

    @staticmethod
    def init(value):
        """Set up an empty literal parser"""
        return LiteralParser(goal=value, so_far='', completed=False, can_continue=True)

    @staticmethod
    def step(state: Union['LiteralParser', None], char: str):
        if not state:
            return None

        if state.goal == state.so_far + char:
            return state.update({
                'so_far': state.goal,
                'completed': True,
                'can_continue': False
            })

        if state.goal.startswith(state.so_far + char):
            return state.update({
                'so_far': state.so_far + char
            })

        return None

    @staticmethod
    def candidates(state: Union['LiteralParser', None]) -> Union[Candidates, list[str]]:
        if not state or state.completed:
            return Candidates.NONE

        return [state.goal[len(state.so_far):]]

class UnionParser(NodeParser):
    """Parser for Unions"""

    branches: list[NodeParser] = field()
    schema = field()
    completed: bool = field()
    can_continue: bool = field()

    @staticmethod
    def init(branches, schema):
        """Set up an empty union parser"""
        return UnionParser(branches=branches, schema=schema, completed=False, can_continue=True)

    @staticmethod
    def step(state: Union['UnionParser', None], char: str):
        if not state:
            return None

        branches = list(filter(lambda b: b is not None, map(lambda b: b.step(b, char), state.branches)))
        if len(branches) == 0:
            return None

        completed = any(b.completed for b in branches)
        can_continue = any(b.can_continue for b in branches)

        return state.update({
            'branches': branches,
            'completed': completed,
            'can_continue': can_continue
        })

    @staticmethod
    def candidates(state: Union['LiteralParser', None]) -> Union[Candidates, list[str]]:
        if not state:
            return Candidates.NONE

        return reduce_candidates(*[b.candidates(b) for b in state.branches])