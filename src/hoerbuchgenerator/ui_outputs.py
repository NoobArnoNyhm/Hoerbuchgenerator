OUTPUT_LIBRARY_HTML = """
<style>
.output-library { color: var(--body-text-color, #222); }
.output-library h2 { font-size: 1.25rem; font-weight: 600; margin-bottom: .75rem; }
.output-library .output-list { display: grid; gap: .75rem; }
.output-library .output-row {
    display: grid; grid-template-columns: minmax(0, 1fr) minmax(220px, 2fr) auto auto;
    align-items: center; gap: .75rem; padding: .75rem;
    border: 1px solid var(--border-color-primary, #aaa); border-radius: .5rem;
}
.output-library .output-name { overflow-wrap: anywhere; min-width: 0; }
.output-library .output-player { min-width: 0; }
.output-library audio { width: 100%; display: block; }
.output-library .output-error { color: var(--error-text-color, #b91c1c); overflow-wrap: anywhere; }
.output-library .output-button {
    cursor: pointer; display: inline-block; text-align: center;
    padding: .5rem .75rem; border: 1px solid var(--border-color-primary, #aaa);
    border-radius: .4rem; color: inherit; background: var(--button-secondary-background-fill, #eee);
    font: inherit; text-decoration: none;
}
.output-library .output-button:focus-visible { outline: 2px solid currentColor; outline-offset: 2px; }
.output-library .output-button:disabled { opacity: .6; cursor: wait; }
.output-library dialog {
    margin: auto; padding: 1.25rem; width: min(32rem, calc(100vw - 2rem));
    max-height: calc(100vh - 2rem); overflow: auto; box-sizing: border-box;
    border: 1px solid var(--border-color-primary, #aaa); border-radius: .5rem;
    color: inherit; background: var(--body-background-fill, #fff);
}
.output-library dialog::backdrop { background: rgb(0 0 0 / 55%); }
.output-library dialog h3 { font-weight: 600; font-size: 1.1rem; }
.output-library dialog p { margin: .75rem 0; overflow-wrap: anywhere; }
.output-library .output-dialog-actions { display: flex; flex-wrap: wrap; gap: .75rem; justify-content: flex-end; }
@media (max-width: 640px) {
    .output-library .output-row { grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); }
    .output-library .output-name, .output-library .output-player { grid-column: 1 / -1; }
}
</style>
<section id="output-library" class="output-library" aria-labelledby="output-library-title">
    <h2 id="output-library-title">Erstellte Hoerbuecher</h2>
    <p class="output-status" role="status">Hoerbuecher werden geladen ...</p>
    <p class="output-empty" hidden>Noch keine Hoerbuecher vorhanden.</p>
    <div class="output-list"></div>
    <dialog aria-labelledby="output-delete-title" aria-describedby="output-delete-description">
        <h3 id="output-delete-title">Hoerbuch endgueltig loeschen?</h3>
        <p id="output-delete-description">
            Die Datei <strong class="output-delete-name"></strong>
            wird unwiderruflich aus Output geloescht. Dies kann nicht rueckgaengig gemacht werden.
        </p>
        <p class="output-dialog-error output-error" role="alert"></p>
        <div class="output-dialog-actions">
            <button type="button" class="output-button output-cancel">Abbrechen</button>
            <button type="button" class="output-button output-confirm">Endgueltig loeschen</button>
        </div>
    </dialog>
</section>
"""

OUTPUT_LIBRARY_REFRESH_JS = """async () => {
    try {
        const library = await import("/assets/output-library.js");
        await library.refresh();
    } catch (error) {
        const status = document.querySelector("#output-library .output-status");
        if (status) {
            status.textContent = "Hoerbuchansicht konnte nicht geladen werden: " + error.message;
            status.setAttribute("role", "alert");
        }
        console.error("Hoerbuchansicht fehlgeschlagen", error);
    }
}"""
