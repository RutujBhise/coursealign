"""
A stand-in for the LangGraph calls this project uses: StateGraph,
add_node, add_edge, add_conditional_edges, and compile().invoke().

WHY IT EXISTS: so the pipeline and its tests run on a machine without
LangGraph installed (like the sandbox this was first built in). When
LangGraph is installed, graph.py always uses the real thing instead.

It copies three LangGraph behaviors, so a wiring mistake fails here too:
a node may only update keys the state declares, a node can't share a name
with a state key, and a run stops after 25 steps.
"""

START, END = "__start__", "__end__"


class StateGraph:
    def __init__(self, state_schema: type):
        self._keys = set(state_schema.__annotations__)
        self._nodes: dict = {}
        self._edges: dict = {}
        self._branches: dict = {}

    def add_node(self, name: str, fn) -> None:
        if name in self._keys:
            raise ValueError(f"'{name}' is already a state key")
        self._nodes[name] = fn

    def add_edge(self, source: str, target: str) -> None:
        self._edges[source] = target

    def add_conditional_edges(self, source: str, route, path_map: dict) -> None:
        self._branches[source] = (route, path_map)

    def compile(self) -> "StateGraph":
        return self

    def invoke(self, state: dict, max_steps: int = 25) -> dict:
        state = dict(state)
        node = self._edges[START]
        for _ in range(max_steps):
            if node == END:
                return state
            update = self._nodes[node](state) or {}
            unknown = set(update) - self._keys
            if unknown:
                raise KeyError(f"node '{node}' returned keys the state doesn't declare: {sorted(unknown)}")
            state.update(update)
            if node in self._branches:
                route, path_map = self._branches[node]
                node = path_map[route(state)]
            else:
                node = self._edges[node]
        raise RuntimeError(f"graph didn't finish within {max_steps} steps")
