const { spawn } = require('child_process');
const path = require('path');

exports.handler = async function(event, context) {
  // Path to the streamlit app
  const appPath = path.join(__dirname, '..', 'dashboard', 'app.py');
  
  // Start the streamlit server
  const streamlit = spawn('streamlit', ['run', appPath]);
  
  // Capture server output
  streamlit.stdout.on('data', (data) => {
    console.log(`stdout: ${data}`);
  });
  
  streamlit.stderr.on('data', (data) => {
    console.error(`stderr: ${data}`);
  });
  
  // Return response
  return {
    statusCode: 200,
    body: 'Streamlit server started'
  };
};
