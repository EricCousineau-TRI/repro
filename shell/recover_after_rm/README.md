# Recovering Files

Attempt to recover post `rm`
<https://linuxconfig.org/how-to-recover-deleted-files-with-foremost-on-linux>

## Trying `foremost`

```sh
./foremost.sh 2>&1 | tee foremost.output.txt
```

## Trying `extundelete`

```sh
./extundelete.sh 2>&1 | tee extundelete.output.txt
```
