from langgraph.graph import StateGraph, START, END
from typing import TypedDict,Literal


class QuadState(TypedDict):
    
    a : int
    b : int
    c : int

    equation : str
    d : float
    result : str

graph = StateGraph(QuadState)


# functions
def show_equation(state : QuadState) -> QuadState:
    coeff_x2 = state['a']
    coeff_x = state['b']
    const = state['c']

    return {
        'equation': f'{coeff_x2}x2 {coeff_x if coeff_x < 0 else f'+ {coeff_x}'}x {const if const < 0 else f'+ {const}'}'
    }

def descriminant(state : QuadState) -> QuadState:
    return {'d' : state['b']**2 - 4*state['a']*state['c']}

def real_roots(state : QuadState) -> QuadState:
    root1 = (-1*state['b'] + state['d']**0.5)/2*state['a']
    root2 = (-1*state['b'] - state['d']**0.5)/2*state['a']

    result = f'The roots are {root1} and {root2}'

    return {'result' : result}

def repeated_roots(state : QuadState) -> QuadState:
    root1 = -1*state['b']/2*state['a']

    result = f'The repeating root is {root1}'

    return {'result' : result}

def img_roots(state : QuadState) -> QuadState:

    result = f'There are no real roots'

    return {'result' : result}


def check_cond(state: QuadState) -> Literal['real_roots', 'repeated_roots', 'img_roots']:
    if state['d'] > 0:
        return 'real_roots'
    elif state['d'] == 0:
        return 'repeated_roots'
    else:
        return 'img_roots'
    


# add node
node1 = graph.add_node('show_equation', show_equation)
node2 = graph.add_node('descriminant', descriminant)
node3 = graph.add_node('real_roots', real_roots)
node3 = graph.add_node('repeated_roots', repeated_roots)
node4 = graph.add_node('img_roots', img_roots)


# add edge
edge1 = graph.add_edge(START, 'show_equation')
edge2 = graph.add_edge('show_equation', 'descriminant')
edge3 = graph.add_conditional_edges('descriminant', check_cond)
edge6 = graph.add_edge('real_roots', END)
edge7 = graph.add_edge('repeated_roots', END)
edge8 = graph.add_edge('img_roots', END)



# compile
workflow = graph.compile()


# result
initial_state = {
    'a':1,
    'b':-4,
    'c':3
}

result = workflow.invoke(initial_state)
print(result)