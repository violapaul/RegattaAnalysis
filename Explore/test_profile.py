

import cProfile
import pstats

command_to_run = ''

cProfile.run(command_to_run, 'profile')

p = pstats.Stats('profile')

p.strip_dirs().sort_stats(-1).print_stats()
p.sort_stats('time').print_stats(10)
p.sort_stats('cumulative').print_stats(30)


def show_all(*args):
    print(args)


