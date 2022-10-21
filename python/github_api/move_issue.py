import base


def main():
    parser = argparse.ArgumentParser()
    base.add_parser(parser)
    parser.add_argument(
        "--from", type=str, required=True,
        help="Form of {owner}/{repo_name}#{issue}.",
    )
    parser.add_argument(
        "--to", type=str, required=True,
        help="Form of {owner}/{repo_name}.",
    )
    args = parser.parse_args()

    gh = base.login(args)

    


assert __name__ == "__main__"
main()
