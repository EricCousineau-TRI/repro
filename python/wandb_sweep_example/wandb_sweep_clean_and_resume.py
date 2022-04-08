import argparse
from subprocess import run, PIPE, STDOUT
from textwrap import dedent, indent

import wandb


def yesno(prompt):
    print(f"{prompt} [y/N]")
    answer = input()
    if answer:
        c = answer[0].lower()
        if c == 'y':
            return True
    return False


def maybe_resume_sweep(sweep):
    # TODO(eric.cousineau): Make this use API instead of CLI.
    sweep_path = f"{sweep.entity}/{sweep.project}/{sweep.id}"
    result = run(
        ["wandb", "sweep", "--resume", sweep_path], stdout=PIPE, stderr=STDOUT, text=True
    )
    print(result.stdout)
    if result.stdout != 0:
        print(dedent(f"""
        WARNING! At time of writing (2022-03-08), `wandb sweep --resume` via CLI may fail.
        https://github.com/wandb/client/issues/3344

        As workaround, go to sweep controls page:
            {sweep.url}/controls
        And manually click "Resume"
        """))  # noqa


def clean_and_resume_sweep(entity, project_name, sweep_id, yes):
    api = wandb.Api()
    sweep_path = f"{entity}/{project_name}/{sweep_id}"
    sweep = api.sweep(path=sweep_path)
    to_delete = []
    for run in sweep.runs:
        if run.state != "finished":
            to_delete.append(run)
    if len(to_delete) > 0:
        print("Will delete following runs:")
        to_delete_txt = "\n".join(
            f"id={run.id}, name={repr(run.name)}, state={run.state}"
            for run in to_delete
        )
        print(indent(to_delete_txt, "  "))
        if not yes and not yesno("Continue?"):
            print("Aborting")
            return
        for run in to_delete:
            run.delete()
    maybe_resume_sweep(sweep)


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", "-e", type=str, required=True)
    parser.add_argument("--project", "-p", type=str, required=True)
    parser.add_argument("-y", "--yes", action="store_true")
    parser.add_argument("sweep_id", type=str)
    args = parser.parse_args(args=argv)
    clean_and_resume_sweep(args.entity, args.project, args.sweep_id, args.yes)


if __name__ == "__main__":
    main()
