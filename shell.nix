{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = with pkgs; [
    (python313.withPackages (ps: with ps; [
      numpy
      pandas
      scikit-learn
      matplotlib
      seaborn
      joblib
    ]))
  ];
}
