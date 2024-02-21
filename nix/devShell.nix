{ mkShell
, stdenv
, python310
}:

let
  unstable = import (fetchTarball "https://releases.nixos.org/nixos/unstable/nixos-22.11pre421761.\
448a599c499/nixexprs.tar.xz") { };
in
{ nixpkgs ? import <nixpkgs> {} }:

nixpkgs.mkShell {
  nativeBuildInputs = with nixpkgs; [
    unstable.python310Packages.poetry
    unstable.python310
  ];

  LD_LIBRARY_PATH = "${nixpkgs.stdenv.cc.cc.lib}/lib";
}