import shlex, subprocess

"""
command_stem = [
"python cs285/scripts/post_process_training_logs.py --prefix MIMICvdl_ --x_tag Train_itr --y_tag {yvar} --save",
"python cs285/scripts/post_process_training_logs.py --prefix pDQNvdl_ --x_tag Train_itr --y_tag {yvar} --baseline_model MIMICvdl_baseline  --save"
]

y = ['Rho','Rho_mort','Rho_train','Rho_mort_train','Training_Loss']

commands = []
for command in command_stem:
    for i in range(len(y)):
        commands.append(command.format(yvar=y[i]))

if __name__ == "__main__":
    for command in commands:
        print(command)
    user_input = None
    while user_input not in ['y', 'n']:
        user_input = input('Run experiment with above commands? (y/n): ')
        user_input = user_input.lower()[:1]
    if user_input == 'n':
        exit(0)
    for command in commands:
        args = shlex.split(command)
        process = subprocess.Popen(args)
        process.wait()
"""

command_stem = [
"python cs285/scripts/post_process_training_logs.py --prefix MIMICCQL_ --x_tag Train_itr --y_tag {yvar} --save",
"python cs285/scripts/post_process_training_logs.py --prefix pCQLv2_ --x_tag Train_itr --y_tag {yvar} --baseline_model MIMICCQL_baseline  --save"
]

y = ['Rho','Rho_mort','Rho_train','Rho_mort_train','Training_Loss']

commands = []
for command in command_stem:
    for i in range(len(y)):
        commands.append(command.format(yvar=y[i]))

if __name__ == "__main__":
    for command in commands:
        print(command)
    user_input = None
    while user_input not in ['y', 'n']:
        user_input = input('Run experiment with above commands? (y/n): ')
        user_input = user_input.lower()[:1]
    if user_input == 'n':
        exit(0)
    for command in commands:
        args = shlex.split(command)
        process = subprocess.Popen(args)
        process.wait()

