CASE_PROMPT = """##Task
{task}
##Interaction trajectory
{trajectory}
"""

CASE_PROMPT_REFLEXION = """##Task
{task}
##Interaction trajectory
{trajectory}
##Feedback
{feedback}
"""

CBR_PROMPT = """You are a helpful assistant to do some scientific experiment in an environment. In the environment, there are several rooms: kitchen, foundry, workshop, bathroom, outside, living room, bedroom, greenhouse, art studio, hallway. You should explore the environment and find the items you need to complete the experiment. You can teleport to any room in one step. All containers in the environment have already been opened, you can directly get items from the containers.

Here are some relevant cases:
{case_prompt}

Here is the task description, and the initial observation by taking action look around:
{task}
Here are all the action templates and their corresponding explanation.
open OBJ: open a container
close OBJ: close a container
activate OBJ: activate a device
deactivate OBJ: deactivate a device
connect OBJ to OBJ: connect electrical components
disconnect OBJ: disconnect electrical components
use OBJ [on OBJ]: use a device/item
look around: describe the current room
examine OBJ: describe an object in detail
look at OBJ: describe a container's contents
read OBJ: read a note or book
move OBJ to OBJ: move an object to a container
pick up OBJ: move an object to the inventory
pour OBJ into OBJ: pour a liquid into a container
mix OBJ: chemically mix a container
teleport to LOC: teleport to a specific room
focus on OBJ: signal intent on a task object
wait: task no action for 10 steps
wait1: task no action for a step
Now it’s your turn to take actions. Please output the action using the action template above. Let's start!
"""


ZERO_SHOT_PROMPT = """You are a helpful assistant to do some scientific experiment in an environment. In the environment, there are several rooms: kitchen, foundry, workshop, bathroom, outside, living room, bedroom, greenhouse, art studio, hallway. You should explore the environment and find the items you need to complete the experiment. You can teleport to any room in one step. All containers in the environment have already been opened, you can directly get items from the containers.
Here is the task description, and the initial observation by taking action look around:
{task}
Here are all the action templates and their corresponding explanation.
open OBJ: open a container
close OBJ: close a container
activate OBJ: activate a device
deactivate OBJ: deactivate a device
connect OBJ to OBJ: connect electrical components
disconnect OBJ: disconnect electrical components
use OBJ [on OBJ]: use a device/item
look around: describe the current room
examine OBJ: describe an object in detail
look at OBJ: describe a container's contents
read OBJ: read a note or book
move OBJ to OBJ: move an object to a container
pick up OBJ: move an object to the inventory
pour OBJ into OBJ: pour a liquid into a container
mix OBJ: chemically mix a container
teleport to LOC: teleport to a specific room
focus on OBJ: signal intent on a task object
wait: task no action for 10 steps
wait1: task no action for a step
Now it’s your turn to take actions. Please output the action using the action template above. Let's start!
"""

SUFFIX_PROMPT = """
Please directly output the action with the format: [Action] action
If you are required to enter a number, please output the action with the format: [Action] number
"""

HISTORY_PROMPT = """[Action] {action}
[Observation] {state}
"""


