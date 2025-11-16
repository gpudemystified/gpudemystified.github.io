/**
 * Output Formatter Utility
 * Formats backend responses for display in output panels
 */

// Guard to prevent multiple loads
if (typeof window.OutputFormatter === 'undefined') {
    window.OutputFormatter = {
        /**
         * Format a backend response with steps
         * @param {Object} result - Backend response object
         * @param {string} result.result - "passed" or "failed"
         * @param {Array} result.steps - Array of step objects
         * @param {string} result.user_output - Program stdout
         * @param {Object} result.metadata - Execution metadata
         * @returns {string} HTML string
         */
        format(result) {
            const isPassed = result.result === "passed";
            
            // Build steps HTML
            const stepsHtml = this.formatSteps(result.steps);
            
            // Build user output section
            const userOutputHtml = this.formatUserOutput(result.user_output);
            
            // Build metadata section
            const metadataHtml = this.formatMetadata(result.metadata);
            
            // Combine everything
            return `<div class="${isPassed ? 'output-success' : 'output-failure'}">
                <div class="output-status">
                    <i class="fas fa-${isPassed ? 'check' : 'times'}-circle"></i>
                    <strong>${isPassed ? 'Success!' : 'Failed'}</strong>
                </div>
                ${stepsHtml}
                ${userOutputHtml}
                ${metadataHtml}
            </div>`;
        },

        /**
         * Format steps array into HTML
         * @param {Array} steps - Array of step objects
         * @returns {string} HTML string
         */
        formatSteps(steps) {
            if (!steps || steps.length === 0) return '';
            
            return steps.map(step => {
                const icon = step.status === 'passed' ? '✓' : 
                            step.status === 'failed' ? '✗' : '⊘';
                const statusClass = step.status === 'passed' ? 'step-passed' : 
                                   step.status === 'failed' ? 'step-failed' : 'step-skipped';
                
                return `<div class="output-step ${statusClass}">
                    <div class="step-header">
                        <span class="step-icon">${icon}</span>
                        <span class="step-name">${this.capitalize(step.name)}</span>
                        <span class="step-status">${step.status}</span>
                    </div>
                    ${step.message ? `<div class="step-message">${this.escapeHtml(step.message)}</div>` : ''}
                    ${step.details ? `<pre class="step-details">${this.escapeHtml(step.details)}</pre>` : ''}
                </div>`;
            }).join('');
        },

        /**
         * Format user output section
         * @param {string} userOutput - Program stdout
         * @returns {string} HTML string
         */
        formatUserOutput(userOutput) {
            if (!userOutput) return '';
            
            return `<div class="user-output-section">
                <div class="section-header">Program Output</div>
                <pre class="output-details">${this.escapeHtml(userOutput)}</pre>
            </div>`;
        },

        /**
         * Format metadata section
         * @param {Object} metadata - Execution metadata
         * @returns {string} HTML string
         */
        formatMetadata(metadata) {
            if (!metadata) return '';
            
            const items = [];
            if (metadata.execution_time_ms) {
                items.push(`⏱️ ${metadata.execution_time_ms}ms`);
            }
            if (metadata.memory_used_mb) {
                items.push(`💾 ${metadata.memory_used_mb}MB`);
            }
            if (metadata.gpu_used) {
                items.push(`🎮 ${metadata.gpu_used}`);
            }
            
            if (items.length === 0) return '';
            
            return `<div class="output-metadata">${items.join(' • ')}</div>`;
        },

        /**
         * Capitalize first letter of string
         * @param {string} str - Input string
         * @returns {string} Capitalized string
         */
        capitalize(str) {
            if (!str) return '';
            return str.charAt(0).toUpperCase() + str.slice(1);
        },

        /**
         * Escape HTML to prevent XSS
         * @param {*} text - Text to escape
         * @returns {string} Escaped HTML string
         */
        escapeHtml(text) {
            if (!text) return '';
            if (typeof text !== 'string') {
                text = JSON.stringify(text, null, 2);
            }
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
    };

    console.log('✓ OutputFormatter loaded');
} else {
    console.warn('⚠ OutputFormatter already loaded, skipping duplicate');
}