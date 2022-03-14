from arguments import get_common_args
from agent import BCAgent


def main():
    args = get_common_args()

    agent = BCAgent(args.input_size, args.output_size, args)
    agent.learn()


if __name__ == '__main__':

    main()
