/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *--------------------------------------------------------------------------------------------*/
module.exports=function(e){return{transformIncoming:e=>"vscode-remote"===e.scheme?{scheme:"file",path:e.path}:"file"===e.scheme?{scheme:"vscode-local",path:e.path}:e,transformOutgoing:o=>"file"===o.scheme?{scheme:"vscode-remote",authority:e,path:o.path}:"vscode-local"===o.scheme?{scheme:"file",path:o.path}:o,transformOutgoingScheme:e=>"file"===e?"vscode-remote":"vscode-local"===e?"file":e}};
//# sourceMappingURL=https://ticino.blob.core.windows.net/sourcemaps/ea3859d4ba2f3e577a159bc91e3074c5d85c0523/core/vs/server/uriTransformer.js.map
