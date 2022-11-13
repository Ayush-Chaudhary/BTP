import numpy as np
from models import env, plot_history, SeqHalf_with_change, PureExploration, config

if __name__ == "__main__":
    # agent = SeqHalf_with_change(config, chng_time=2048)
    agent = PureExploration(config, chng_time=512)
    history = agent.act()
    plot_history(history)