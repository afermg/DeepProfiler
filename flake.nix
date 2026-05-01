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
            # cudaSupport=true selects nixpkgs' python311-tensorflow-gpu-2.13.0
            # derivation (a separate, CUDA-linked TF wheel — not the same store
            # path as the CPU-only python311-tensorflow-2.13.0). This pulls in
            # cuda-merged-11.8 + cudnn-merged + UCC/NCCL/NVSHMEM, which on a
            # cold cache compiles UCC/NCCL from source for ~30+ min. Required
            # for GPU runtime, so we keep it on.
            cudaSupport = true;
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
              # tf-keras' wheel METADATA lists `tensorflow` as a runtime dep,
              # but with cudaSupport=true we ship `tensorflow-gpu` (different
              # package name). Skip the runtime-deps check to avoid the
              # "tensorflow not installed" failure — the import path works at
              # runtime since both wheels expose the same `tensorflow` module.
              (pp.tf-keras.overridePythonAttrs (_: { dontCheckRuntimeDeps = true; }))
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
                # TF 2.13 expects standalone keras at runtime; tf-keras 2.17
                # matches the API and works with TF_USE_LEGACY_KERAS=1.
                # tf-keras' wheel METADATA lists `tensorflow` as a runtime dep,
              # but with cudaSupport=true we ship `tensorflow-gpu` (different
              # package name). Skip the runtime-deps check to avoid the
              # "tensorflow not installed" failure — the import path works at
              # runtime since both wheels expose the same `tensorflow` module.
              (pp.tf-keras.overridePythonAttrs (_: { dontCheckRuntimeDeps = true; }))
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
