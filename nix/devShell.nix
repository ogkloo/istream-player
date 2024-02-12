{ mkShell
, stdenv
, python310
}:

let
  # CHANGEME define the list of Python packages to pull from nixpkgs
  python-env = python310.withPackages (pp: with pp;
    # for example
    # [ pyspark numpy ]
    [ numpy ]
  );
in
mkShell {
  # CHANGEME add other requirements
  buildInputs =
    # for example
    # [ jdk8 ]
    []
    # injects the Python base
    ++ [ python-env ];

  NIX_PYTHONPATH = "${python-env}/${python-env.sitePackages}";

  shellHook = ''
    export LD_LIBRARY_PATH=${stdenv.cc.cc.lib}/lib/
    if [[ ! -d .venv ]]; then
      echo "No virtual env found at ./.venv, creating a new virtual env linked to the Python site defined with Nix"
      ${python-env}/bin/python -m venv .venv
      cp ${builtins.toString ./sitecustomize.py} .venv/lib/python*/site-packages/
    fi
    source .venv/bin/activate
    echo "Nix development shell loaded."
  '';
}
