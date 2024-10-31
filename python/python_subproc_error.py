import subprocess


_process_error_fmt = """
Command '{cmd}' died with {returncode}.
stdout:

{stdout}

stderr:

{stderr}
"""


class CalledProcessErrorBetterFormat(subprocess.SubprocessError):
    """
    In at least Python 3.10, CalledProcessError.__str__() does not include
    captured stdout or stderr in its formatted output.
    
    This wraps the instance with __repr__() to include it.
    """

    def __init__(self, e: subprocess.CalledProcessError):
        self.e = e
    
    def __repr__(self):
        return _process_error_fmt.format(
            cmd=self.e.cmd,
            returncode=self.e.returncode,
            stdout=str(self.e.output).rstrip(),
            stderr=str(self.e.stderr).rstrip(),
        )
      
    def __str__(self):
        return repr(self)


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
  except subprocess.CalledProcessError as e:
    raise CalledProcessErrorBetterFormat(e)

if __name__ == "__main__":
    main()

"""
Command 'echo stdout text; echo stderr text >&2; exit 1' returned non-zero exit status 1.
stdout text
stderr text
"""
