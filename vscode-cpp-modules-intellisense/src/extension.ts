import * as cp from "node:child_process";
import * as fs from "node:fs/promises";
import * as path from "node:path";
import * as vscode from "vscode";
import {
  LanguageClient,
  LanguageClientOptions,
  ServerOptions
} from "vscode-languageclient/node";

let client: LanguageClient | undefined;

const extensionId = "cppModulesIntellisense";

export async function activate(context: vscode.ExtensionContext): Promise<void> {
  context.subscriptions.push(
    vscode.commands.registerCommand(
      `${extensionId}.restartLanguageServer`,
      async () => {
        await restartClient(context);
        void vscode.window.showInformationMessage("C++ Modules IntelliSense restarted clangd.");
      }
    ),
    vscode.commands.registerCommand(
      `${extensionId}.configureWorkspaceForModules`,
      async () => {
        await configureWorkspaceForModules();
      }
    ),
    vscode.commands.registerCommand(
      `${extensionId}.createClangdConfig`,
      async () => {
        await createClangdTemplate();
      }
    )
  );

  await startClient(context);
  void checkClangdVersion();
}

export async function deactivate(): Promise<void> {
  if (client) {
    await client.stop();
    client = undefined;
  }
}

async function restartClient(context: vscode.ExtensionContext): Promise<void> {
  if (client) {
    await client.stop();
    client = undefined;
  }

  await startClient(context);
}

async function startClient(context: vscode.ExtensionContext): Promise<void> {
  const config = vscode.workspace.getConfiguration(extensionId);
  const clangdPath = config.get<string>("clangd.path", "clangd");
  const args = [
    ...config.get<string[]>("clangd.arguments", []),
    ...getOptionalQueryDriverArg(config)
  ];

  const serverOptions: ServerOptions = {
    command: clangdPath,
    args,
    transport: 0
  };

  const clientOptions: LanguageClientOptions = {
    documentSelector: [
      { scheme: "file", language: "c" },
      { scheme: "file", language: "cpp" },
      { scheme: "file", language: "cuda" },
      { scheme: "file", language: "cuda-cpp" },
      { scheme: "file", language: "objective-c" },
      { scheme: "file", language: "objective-cpp" }
    ],
    outputChannel: vscode.window.createOutputChannel("C++ Modules IntelliSense"),
    initializationOptions: {
      compilationDatabasePath: emptyToUndefined(
        config.get<string>("clangd.compilationDatabasePath", "")
      ),
      fallbackFlags: getFallbackFlags(config)
    }
  };

  client = new LanguageClient(
    extensionId,
    "C++ Modules IntelliSense",
    serverOptions,
    clientOptions
  );

  context.subscriptions.push(
    new vscode.Disposable(() => {
      if (client) {
        void client.stop();
      }
    })
  );

  await client.start();
}

function getOptionalQueryDriverArg(
  config: vscode.WorkspaceConfiguration
): string[] {
  const queryDriver = config.get<string>("clangd.queryDriver", "").trim();
  return queryDriver ? [`--query-driver=${queryDriver}`] : [];
}

function getFallbackFlags(config: vscode.WorkspaceConfiguration): string[] {
  const configured = config.get<string[]>("clangd.fallbackFlags", []);
  if (configured.length > 0) {
    return configured;
  }

  return [`-std=${config.get<string>("modules.languageStandard", "c++23")}`];
}

async function configureWorkspaceForModules(): Promise<void> {
  if (!vscode.workspace.workspaceFolders?.length) {
    void vscode.window.showErrorMessage("Open a workspace folder before configuring module file associations.");
    return;
  }

  const settings = vscode.workspace.getConfiguration("files");
  const existing = settings.get<Record<string, string>>("associations", {});
  const extensions = vscode.workspace
    .getConfiguration(extensionId)
    .get<string[]>("modules.fileExtensions", []);

  const updated = { ...existing };
  for (const pattern of extensions) {
    updated[pattern] = "cpp";
  }

  await settings.update(
    "associations",
    updated,
    vscode.ConfigurationTarget.Workspace
  );

  void vscode.window.showInformationMessage("Workspace configured so module file extensions open as C++.");
}

async function createClangdTemplate(): Promise<void> {
  const folder = vscode.workspace.workspaceFolders?.[0];
  if (!folder) {
    void vscode.window.showErrorMessage("Open a workspace folder before creating a .clangd file.");
    return;
  }

  const clangdPath = path.join(folder.uri.fsPath, ".clangd");
  const exists = await fileExists(clangdPath);
  if (exists) {
    const choice = await vscode.window.showWarningMessage(
      ".clangd already exists. Overwrite it?",
      "Overwrite",
      "Cancel"
    );

    if (choice !== "Overwrite") {
      return;
    }
  }

  const config = vscode.workspace.getConfiguration(extensionId);
  const languageStandard = config.get<string>("modules.languageStandard", "c++23");
  const compilationDatabasePath = config.get<string>("clangd.compilationDatabasePath", "").trim();
  const suppressDiagnostics = config.get<boolean>("modules.suppressDiagnostics", false);

  const template = buildClangdTemplate(
    languageStandard,
    compilationDatabasePath,
    suppressDiagnostics
  );
  await fs.writeFile(clangdPath, template, "utf8");

  const document = await vscode.workspace.openTextDocument(clangdPath);
  await vscode.window.showTextDocument(document);
  void vscode.window.showInformationMessage("Wrote a .clangd template for modern C++ modules.");
}

function buildClangdTemplate(
  languageStandard: string,
  compilationDatabasePath: string,
  suppressDiagnostics: boolean
): string {
  const compilationDatabase = compilationDatabasePath
    ? `  CompilationDatabase: ${compilationDatabasePath}\n`
    : "";

  const lines = [
    "# Generated by C++ Modules IntelliSense",
    "---",
    "CompileFlags:",
    ...toIndentedLines(compilationDatabase),
    "  Remove: [-std=*]",
    `  Add: [-std=${languageStandard}]`,
    "---",
    "If:",
    "  PathMatch: .*\\.(cppm|ixx|mpp|mxx|ccm|cxxm)$",
    "CompileFlags:",
    "  Remove: [-x]",
    "  Add: [-xc++-module]"
  ];

  if (suppressDiagnostics) {
    lines.push("Diagnostics:");
    lines.push("  Suppress: '*'");
  }

  lines.push("");
  return lines.join("\n");
}

function toIndentedLines(value: string): string[] {
  return value
    .split("\n")
    .filter((line) => line.length > 0);
}

async function checkClangdVersion(): Promise<void> {
  const config = vscode.workspace.getConfiguration(extensionId);
  const clangdPath = config.get<string>("clangd.path", "clangd");
  const result = cp.spawnSync(clangdPath, ["--version"], { encoding: "utf8" });

  if (result.error || result.status !== 0) {
    void vscode.window.showWarningMessage(
      "clangd was not found. Set cppModulesIntellisense.clangd.path or install clangd."
    );
    return;
  }

  const output = `${result.stdout ?? ""}\n${result.stderr ?? ""}`;
  const match = output.match(/version\s+(\d+)/i);
  if (!match) {
    return;
  }

  const major = Number.parseInt(match[1], 10);
  if (Number.isNaN(major) || major >= 18) {
    return;
  }

  void vscode.window.showWarningMessage(
    "clangd < 18 detected. For C++20/23 modules, use a newer clangd if possible."
  );
}

async function fileExists(filePath: string): Promise<boolean> {
  try {
    await fs.access(filePath);
    return true;
  } catch {
    return false;
  }
}

function emptyToUndefined(value: string): string | undefined {
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : undefined;
}
