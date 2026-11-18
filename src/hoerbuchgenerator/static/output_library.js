let current = null;

async function request(url, options = {}) {
    const response = await fetch(url, {cache: "no-store", ...options});
    if (!response.ok) {
        const body = await response.json();
        const error = new Error(body.detail?.message || `HTTP ${response.status}`);
        error.status = response.status;
        throw error;
    }
    return response.status === 204 ? null : response.json();
}

function createLibrary(root) {
    const list = root.querySelector(".output-list");
    const status = root.querySelector(".output-status");
    const empty = root.querySelector(".output-empty");
    const dialog = root.querySelector("dialog");
    const dialogError = root.querySelector(".output-dialog-error");
    const cancel = root.querySelector(".output-cancel");
    const confirm = root.querySelector(".output-confirm");
    const rows = new Map();
    let selected = null;
    let deleting = false;
    let inFlight = null;

    function showError(message) {
        status.textContent = message;
        status.classList.add("output-error");
        status.setAttribute("role", "alert");
    }

    function createRow(file) {
        const row = document.createElement("div");
        row.className = "output-row";
        const name = document.createElement("span");
        name.className = "output-name";
        name.textContent = file.name;
        const player = document.createElement("div");
        player.className = "output-player";
        const audio = document.createElement("audio");
        audio.controls = true;
        audio.preload = "none";
        audio.src = file.url;
        audio.setAttribute("aria-label", `Hoerbuch ${file.name}`);
        const playbackError = document.createElement("p");
        playbackError.className = "output-error";
        playbackError.setAttribute("role", "alert");
        playbackError.hidden = true;
        audio.addEventListener("error", () => {
            playbackError.textContent = "Audio kann nicht wiedergegeben werden. Download und Loeschen bleiben verfuegbar.";
            playbackError.hidden = false;
        });
        player.append(audio, playbackError);
        const download = document.createElement("a");
        download.className = "output-button";
        download.textContent = "Herunterladen";
        download.href = file.url;
        download.download = file.name;
        download.setAttribute("aria-label", `${file.name} herunterladen`);
        const remove = document.createElement("button");
        remove.type = "button";
        remove.className = "output-button";
        remove.textContent = "Loeschen";
        remove.setAttribute("aria-label", `${file.name} loeschen`);
        remove.addEventListener("click", () => {
            selected = Object.freeze({...file});
            root.querySelector(".output-delete-name").textContent = selected.name;
            dialogError.textContent = "";
            confirm.disabled = false;
            dialog.showModal();
            cancel.focus();
        });
        row.append(name, player, download, remove);
        return row;
    }

    function reconcile(files) {
        const keys = new Set(files.map(file => `${file.name}:${file.version}`));
        for (const [key, row] of rows) {
            if (!keys.has(key)) {
                const audio = row.querySelector("audio");
                audio.pause();
                audio.removeAttribute("src");
                audio.load();
                row.remove();
                rows.delete(key);
            }
        }
        files.forEach((file, index) => {
            const key = `${file.name}:${file.version}`;
            let row = rows.get(key);
            if (!row) {
                row = createRow(file);
                rows.set(key, row);
            }
            // Insert new rows without detaching unchanged, possibly playing media.
            if (list.children[index] !== row) {
                list.insertBefore(row, list.children[index] || null);
            }
        });
        empty.hidden = files.length !== 0;
    }

    async function update() {
        try {
            const body = await request("/api/v1/output");
            reconcile(body.files);
            status.textContent = "";
            status.classList.remove("output-error");
            status.setAttribute("role", "status");
        } catch (error) {
            showError(`Hoerbuchbestand konnte nicht aktualisiert werden: ${error.message}`);
        }
    }

    async function refresh() {
        if (!inFlight) {
            inFlight = update().finally(() => { inFlight = null; });
        }
        await inFlight;
    }

    cancel.addEventListener("click", () => dialog.close());
    dialog.addEventListener("keydown", event => {
        if (event.key === "Escape") {
            event.preventDefault();
            if (!deleting) dialog.close();
        }
    });
    dialog.addEventListener("cancel", event => {
        if (deleting) event.preventDefault();
    });
    dialog.addEventListener("close", () => {
        selected = null;
        dialogError.textContent = "";
        confirm.disabled = false;
    });
    confirm.addEventListener("click", async () => {
        if (!selected || deleting) return;
        const target = selected;
        deleting = true;
        confirm.disabled = true;
        cancel.disabled = true;
        let removed = false;
        try {
            await request(`/api/v1/output/${encodeURIComponent(target.name)}`, {
                method: "DELETE",
                headers: {"If-Match": `"${target.version}"`},
            });
            removed = true;
            dialog.close();
        } catch (error) {
            dialogError.textContent = `Loeschen fehlgeschlagen: ${error.message}`;
            // A stale selection must be closed and selected anew, never silently retargeted.
        } finally {
            deleting = false;
            cancel.disabled = false;
            if (!removed) confirm.disabled = true;
            await inFlight;
            await refresh();
        }
    });
    return {root, refresh};
}

export async function refresh() {
    const root = document.querySelector("#output-library");
    if (!root) return;
    if (!current || current.root !== root) current = createLibrary(root);
    await current.refresh();
}
