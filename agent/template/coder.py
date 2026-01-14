from string import Template

# One Shot 
INIT_CODER_TEMPLATE = Template("""
You write custom CUDA kernels to replace the pytorch operators in the given architecture to get speedups. \n
                               
You have complete freedom to choose the set of operators you want to replace. You may make the decision to replace some operators with custom CUDA kernels and leave others unchanged. You may replace multiple operators with custom implementations, consider operator fusion opportunities (combining multiple operators into a single kernel, for example, combining matmul+relu), or algorithmic changes (such as online softmax). You are only limited by your imagination.\n

Here's an example to show you the syntax of inline embedding custom CUDA operators in torch: The example given architecture is: \n
                               
$example_source_code \n

The example new arch with custom CUDA kernels looks like this: \n
                               
$example_new_code \n
                               
You are given the following architecture: \n

$source_code \n
                        
Here is some hints to help you to optimize the architecture: \n

$hints \n
                                                          
Optimize the architecture named Model with custom CUDA operators! Name your optimized output architecture ModelNew. Output the new code in codeblocks. Please generate real code, NOT pseudocode, make sure the code compiles and is fully functional. Just output the new model code, no other text, and NO testing code! \n                           
""")