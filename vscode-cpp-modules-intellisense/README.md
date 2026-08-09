# C++ Modules IntelliSense

`C++ Modules IntelliSense` is a VS Code extension that starts `clangd` as a language server and adds the missing workspace setup that modern C++ projects usually need for modules:

- module file extensions associated with the built-in `cpp` language
- a generated `.clangd` template for C++20/23/26 module units
- optional suppression of diagnostics in module files
- clangd fallback flags when a compilation database is missing
- clangd attachment for CUDA editors such as `cuda-cpp`

## Why this design

This extension uses a standard VS Code language client and delegates parsing, completion, diagnostics, definitions, references, and semantic analysis to `clangd`.

That is the practical route for modern C++ because modules support depends on the real compiler command line:

- `clangd` consumes compile commands such as `-std=...`, `-I`, and `-x`
- standard C++ modules are enabled by Clang automatically with `-std=c++20` or newer

## Requirements

- VS Code
- `clangd` on `PATH`, or set `cppModulesIntellisense.clangd.path`
- ideally a `compile_commands.json`

## Commands

- `C++ Modules IntelliSense: Restart Language Server`
- `C++ Modules IntelliSense: Configure Workspace for Modules`
- `C++ Modules IntelliSense: Create .clangd Template`

## How to use it

1. Install the `.vsix` package in VS Code.
2. Make sure `clangd` is installed, or set `cppModulesIntellisense.clangd.path`.
3. Open your C++ project folder.
4. Run `C++ Modules IntelliSense: Configure Workspace for Modules`.
5. Run `C++ Modules IntelliSense: Create .clangd Template`.
6. Run `C++ Modules IntelliSense: Restart Language Server`.

After that:

- `*.cppm`, `*.ixx`, `*.mpp`, `*.mxx`, `*.ccm`, and `*.cxxm` open as C++
- clangd parses those files with `-xc++-module`
- module-file diagnostics stay enabled by default
- CUDA files opened as `cuda-cpp` can still be handled by the same clangd client

## Optional diagnostic suppression

The setting `cppModulesIntellisense.modules.suppressDiagnostics` defaults to `false`.

If you turn it on, the generated `.clangd` includes a fragment for module files like:

```yaml
If:
  PathMatch: .*\.(cppm|ixx|mpp|mxx|ccm|cxxm)$
CompileFlags:
  Remove: [-x]
  Add: [-xc++-module]
Diagnostics:
  Suppress: '*'
```

This removes clangd diagnostics from module units only. Other C, C++, and CUDA files still keep diagnostics.

If you need a quiet editing mode for module units, set `cppModulesIntellisense.modules.suppressDiagnostics` to `true` and regenerate `.clangd`.

## Recommended workflow

1. Install a recent `clangd`.
2. Open the project in VS Code.
3. Run `C++ Modules IntelliSense: Configure Workspace for Modules`.
4. Run `C++ Modules IntelliSense: Create .clangd Template`.
5. Make sure your build exports `compile_commands.json`.

## Notes on modules

Module IntelliSense is only as accurate as the compile commands that describe your build. If your project compiles modules with custom flags or generates BMIs in specific places, those details must be visible to `clangd` through the compilation database or `.clangd`.
