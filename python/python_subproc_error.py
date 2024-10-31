import subprocess

def main():
  try:
    subprocess.run(
      "echo stdout text; echo stderr text >&2; exit 1",
      shell=True,
      text=True,
      stdout=subprocess.PIPE,
      stderr=subprocess.STDOUT,
      check=True,
    )
  except Exception as e:
    print(e)
    print(e.stdout.strip())
    if e.stderr is not None:
      print(e.stderr)

if __name__ == "__main__":
    main()

"""
Command 'echo stdout text; echo stderr text >&2; exit 1' returned non-zero exit status 1.
stdout text
stderr text
"""
