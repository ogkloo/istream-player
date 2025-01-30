{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  nativeBuildInputs = [ pkgs.openssl ];
  LD_LIBRARY_PATH = "${pkgs.stdenv.cc.cc.lib}/lib";
}
