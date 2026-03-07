// Global utility functions (e.g., formatDate, etc)

// --- Helper Functions ---

// FUNCTION 1:-
/**
 * Checks if a string contains any of the specified keywords.
 * @param {string} str - The string to check (e.g., array[0]).
 * @returns {string} - The original string if matched, otherwise "".
 */
export function getPartialMatch(str) {
    // this function can be used to find a 'unique key' for the table, dynamically.
    if (!str) return "";

    // Note:
    // .some(): Returns "true" as soon as one element in the keywords array satisfies the condition.
    // .includes(): Determines whether one string can be found within another, returning a "boolean".

    // Define the keywords to look for
    const keywords = ["id", "serial", "sl", "no.", "index"];

    // Perform case-insensitive search
    const isMatch = keywords.some(key =>
        str.toLowerCase().includes(key.toLowerCase())
    );
    return isMatch ? str : "";
}
// --- Usage Examples ---
// const SL1 = getPartialMatch("sl no."); // Returns "sl no."
// const SL2 = getPartialMatch("__id");   // Returns "__id"
// const SL3 = getPartialMatch("key");    // Returns ""


// FUNCTION 2:-