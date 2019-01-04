sample-command() {
    echo "arguments: $@"
}
__sample-command() {
    # Simulate loading for the first time
    if [[ -z ${sample_command_init-} ]]; then
        echo -e "\n"
        cat <<EOF
Please wait while sample-command initializes autocomplete.
This will only happen when the cache is invalidatede.
If this happens too often, please consider setting:
    export SAMPLE_COMMAND_COMPLETION_USE_SLOW=false
in ~/.bash_completion, ~/.bash_aliases, or ~/.bashrc.
This will provide a quicker, but less accurate, completion mechanism.
EOF
        for i in $(seq 10); do
            sleep 0.1
            echo -n "."
        done
        echo -e "\n[ Done ]"
        sample_command_init=1
        # Restore prompt (when single-tab is pressed... will duplicate on double-tab)
        show-prompt
        echo -n "${COMP_WORDS[@]}"
    fi
    local cur=${COMP_WORDS[COMP_CWORD]}
    COMPREPLY=($(compgen -W "subcommand other_subcommand help" -- $cur))
}
complete -F __sample-command "sample-command"

show-prompt() {
    # Simpler than: http://stackoverflow.com/questions/22322879/how-to-print-current-bash-prompt
    # (May be less robust)
    eval 'echo -en "'$PS1'"' | sed -e 's#\\\[##g' -e 's#\\\]##g'
}

<<COMMENT
Example output:

wip(feature/mosek_license_lock)$ sample-command <TAB>

Please wait while sample-command initializes autocomplete.
This will only happen when the cache is invalidatede.
If this happens too often, please consider setting:
    export SAMPLE_COMMAND_COMPLETION_USE_SLOW=false
in ~/.bash_completion, ~/.bash_aliases, or ~/.bashrc.
This will provide a quicker, but less accurate, completion mechanism.
..........
[ Done ]
wip(feature/mosek_license_lock)$ sample-command <TAB>
help              other_subcommand  subcommand

COMMENT
