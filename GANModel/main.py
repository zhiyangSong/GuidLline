from arguments import get_common_args
from agent import GANAgent


def main():
    args = get_common_args()

    agent = GANAgent(args.input_size, args.output_size, args)
    agent.train()


if __name__ == '__main__':

    main()
