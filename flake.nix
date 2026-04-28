{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.11";
    systems.url = "github:nix-systems/default";
    flake-utils.url = "github:numtide/flake-utils";
    flake-utils.inputs.systems.follows = "systems";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      systems,
      ...
    }@inputs:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          system = system;
          config = {
            allowUnfree = true;
            # tensorflow 2.13's CUDA 11.8 stack rebuilds nccl/ucc from source under
            # nixos-24.11, which can take >30min on contended hosts. The
            # python311-tensorflow-2.13.0 store path is the same regardless of
            # cudaSupport (CPU-only at the python wheel layer here), so leave
            # cudaSupport off to avoid pulling cuda-merged-11.8 + cudnn-merged
            # into the wrapper env. Re-enable if you need the GPU runtime libs.
            cudaSupport = false;
          };
        };
      in
      with pkgs;
      rec {
        # tensorflow 2.13 in nixos-24.11 only supports up to python3.11.
        pythonForServer = pkgs.python311;

        apps.default =
          let
            python_with_pkgs = pythonForServer.withPackages (pp: [
              packages.nahual
              pp.tensorflow
              # TF 2.13 in nixos-24.11 expects standalone keras at runtime; the
              # tf-keras 2.17 package matches TF 2.13's compat surface when
              # TF_USE_LEGACY_KERAS=1 is set in the runner.
              pp.tf-keras
              pp.numpy
              pp.pillow
              pp.trio
            ]);
            runServer = pkgs.writeScriptBin "runserver.sh" ''
              #!${pkgs.bash}/bin/bash
              # TF 2.13 in nixos-24.11 expects standalone keras at runtime;
              # using tf-keras 2.17 in legacy mode keeps the API working.
              export TF_USE_LEGACY_KERAS=1
              ${python_with_pkgs}/bin/python ${self}/server.py ''${@:-"ipc:///tmp/deepprofiler.ipc"}
            '';
          in
          {
            type = "app";
            program = "${runServer}/bin/runserver.sh";
          };

        packages = {
          # Build pynng locally for python3.11 (tensorflow 2.13's interpreter).
          pynng = pythonForServer.pkgs.callPackage ./nix/pynng.nix { };
          nahual = pythonForServer.pkgs.callPackage ./nix/nahual.nix {
            pynng = packages.pynng;
          };
        };

        devShells = {
          default =
            let
              python_with_pkgs = pythonForServer.withPackages (pp: [
                packages.nahual
                pp.tensorflow
                pp.numpy
                pp.pillow
                pp.trio
                pp.scikit-image
                pp.scikit-learn
                pp.pyyaml
              ]);
            in
            mkShell {
              packages = [
                python_with_pkgs
                pkgs.cudaPackages.cudatoolkit
              ];
              shellHook = ''
                export PYTHONPATH=${python_with_pkgs}/${python_with_pkgs.sitePackages}:$PYTHONPATH
              '';
            };
        };
      }
    );
}
