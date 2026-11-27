// electron-builder hook (afterPack). Signs the finished .app ad hoc, with no certificate.
//
// Without a certificate, electron-builder skips signing and leaves only the linker's
// signature on the Electron binary. macOS then reports the app as "damaged" when it is
// downloaded. Signing the whole bundle with `codesign --sign -` seals the resources, so
// Gatekeeper accepts the app as an unidentified developer's app (right-click > Open).
const { execFileSync } = require('child_process');
const path = require('path');

exports.default = async function adHocSign(context) {
  if (context.electronPlatformName !== 'darwin') return;
  const appName = `${context.packager.appInfo.productFilename}.app`;
  const appPath = path.join(context.appOutDir, appName);
  execFileSync('codesign', ['--force', '--deep', '--sign', '-', appPath], { stdio: 'inherit' });
  execFileSync('codesign', ['--verify', '--deep', '--strict', appPath], { stdio: 'inherit' });
};
