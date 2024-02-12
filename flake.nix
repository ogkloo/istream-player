{
  inputs.flake-utils.url = "github:numtide/flake-utils";

  outputs = { self, nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        packageName = "istream-player";
      in
      {
        #defaultPackage = self.packages.${system}.${packageName};
        devShell = pkgs.callPackage ./nix/devShell.nix { };
      }
    );
}
