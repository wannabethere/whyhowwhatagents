from typing import Union, Optional, List
import operator
from typing import (
    Any,
    Awaitable,
    Callable,
    Hashable,
    Literal,
    NamedTuple,
    Annotated,
    Optional,
    Sequence,
    Union,
    cast,
    get_args,
    get_origin,
    get_type_hints,
    overload,
    List
)
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.runnables import Runnable
from langchain_core.runnables.base import RunnableLike
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, StateGraph, START
from langchain_core.tools import tool
import functools
import operator
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    ToolMessage,
    AIMessage
)
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI


class WFNode:
    node: Union[str, RunnableLike]
    system_message: str
    nodetype: str= 'agent'
    tools: Optional[List[Any]]
    action: Optional[RunnableLike] = None
    metadata: Optional[dict[str, Any]] = None
    name: str
        
class WFEdge:
   start_key: str
   end_key: str

class WFConditionalEdge:
    start_key: str
    path: Union[
            Callable[..., Union[Hashable, list[Hashable]]],
            Callable[..., Awaitable[Union[Hashable, list[Hashable]]]],
            Runnable[Any, Union[Hashable, list[Hashable]]],
        ]
    path_map: Optional[Union[dict[Hashable, str], list[str]]] = None
    then: Optional[str] = None

class BaseAgentState:
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: str



def build_worklow_graph(nodes:List[WFNode],edges:List[WFEdge],cedges:List[WFConditionalEdge], workflow):
    for node in nodes:
        print(node.name)
        workflow.add_node(node.name, node.node)

    for edge in edges:
        print(edge.start_key)
        workflow.add_edge(edge.start_key,edge.end_key)
    print(len(cedges))
    for conditional_edge in cedges:  
        print(conditional_edge.start_key)
        workflow.add_conditional_edges(
            conditional_edge.start_key,
            conditional_edge.path,
            conditional_edge.path_map,
            conditional_edge.then
        )
    graph = workflow.compile()
    return graph
