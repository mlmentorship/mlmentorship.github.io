export const WORKBOOK_UNLOCK_KEY = 'mlmentorship:workbook:email-unlocked:v1';

let unlockedForVisit = false;

export function unlockWorkbookEmail(): void {
	unlockedForVisit = true;
	try { window.localStorage.setItem(WORKBOOK_UNLOCK_KEY, '1'); } catch {}
}

export function isWorkbookEmailUnlocked(): boolean {
	if (unlockedForVisit) return true;
	try { return window.localStorage.getItem(WORKBOOK_UNLOCK_KEY) === '1'; } catch { return false; }
}
