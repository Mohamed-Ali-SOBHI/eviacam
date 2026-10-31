// SPDX-License-Identifier: GPL-3.0-or-later
#include "../keyboardcode.h"
#include <ApplicationServices/ApplicationServices.h>
#include <Carbon/Carbon.h>
#include <cstdlib>

namespace {
const int functionKeys[] = {kVK_F1,kVK_F2,kVK_F3,kVK_F4,kVK_F5,kVK_F6,kVK_F7,kVK_F8,kVK_F9,kVK_F10,kVK_F11,kVK_F12,kVK_F13,kVK_F14,kVK_F15,kVK_F16,kVK_F17,kVK_F18,kVK_F19,kVK_F20};
wxString CharacterForKey(int key, UInt32 modifiers=0) {
    TISInputSourceRef source = TISCopyCurrentKeyboardLayoutInputSource();
    if (!source) return wxString();
    CFDataRef data = (CFDataRef)TISGetInputSourceProperty(source, kTISPropertyUnicodeKeyLayoutData);
    UniChar text[4] = {}; UniCharCount length = 0; UInt32 dead = 0;
    if (data) UCKeyTranslate((const UCKeyboardLayout*)CFDataGetBytePtr(data), key,
        kUCKeyActionDisplay, modifiers, LMGetKbdType(), kUCKeyTranslateNoDeadKeysBit,
        &dead, 4, &length, text);
    CFRelease(source);
    wxString result;
    for (UniCharCount i=0; i<length; ++i) result += wxUniChar(text[i]);
    return result;
}
}
// Native macOS scan codes include zero (A); store scan+1 so zero remains invalid.
wxString KeyboardCode::GetName() const {
    if (!is_valud()) return wxString();
    const int key = get_native()-1;
    for (int i=0; i<20; ++i) if (functionKeys[i] == key) return wxString::Format("F%d",i+1);
    switch (key) {
        case kVK_LeftArrow: return "Left"; case kVK_RightArrow: return "Right";
        case kVK_UpArrow: return "Up"; case kVK_DownArrow: return "Down";
        case kVK_Escape: return "Escape"; case kVK_Return: return "Return";
        case kVK_Delete: return "Backspace"; case kVK_Space: return "Space";
        case kVK_Tab: return "Tab";
        default: return CharacterForKey(key).Upper();
    }
}
void KeyboardCode::SendKey() {
    const char* replay = std::getenv("EVIACAM_REPLAY");
    if ((replay && *replay) || !is_valud() || !AXIsProcessTrusted()) return;
    for (bool down : {true, false}) {
        CGEventRef event = CGEventCreateKeyboardEvent(nullptr, get_native()-1, down);
        if (event) { CGEventPost(kCGHIDEventTap,event); CFRelease(event); }
    }
}
KeyboardCode KeyboardCode::ReadKeyCode() {
    if (!CGPreflightListenEventAccess()) return KeyboardCode();
    for (int key=0; key<128; ++key)
        if (CGEventSourceKeyState(kCGEventSourceStateCombinedSessionState,key)) return FromScanCode(key);
    return KeyboardCode();
}
KeyboardCode KeyboardCode::FromChar(char c) {
    for (int key=0; key<128; ++key) {
        const wxString value = CharacterForKey(key);
        if (value.length()==1 && value.Lower()==wxString(c).Lower()) return FromScanCode(key);
    }
    return KeyboardCode();
}
KeyboardCode KeyboardCode::FromScanCode(int code) { return (code>=0 && code<128) ? KeyboardCode(code+1) : KeyboardCode(); }
KeyboardCode KeyboardCode::FromWXK(wxKeyCode key) {
    if (key>=WXK_F1 && key<=WXK_F20) return FromScanCode(functionKeys[key-WXK_F1]);
    switch (key) {
        case WXK_LEFT: return FromScanCode(kVK_LeftArrow);
        case WXK_RIGHT: return FromScanCode(kVK_RightArrow);
        case WXK_UP: return FromScanCode(kVK_UpArrow);
        case WXK_DOWN: return FromScanCode(kVK_DownArrow);
        case WXK_BACK: return FromScanCode(kVK_Delete);
        case WXK_DELETE: return FromScanCode(kVK_ForwardDelete);
        case WXK_RETURN: return FromScanCode(kVK_Return);
        case WXK_TAB: return FromScanCode(kVK_Tab);
        case WXK_ESCAPE: return FromScanCode(kVK_Escape);
        case WXK_HOME: return FromScanCode(kVK_Home);
        case WXK_END: return FromScanCode(kVK_End);
        case WXK_PAGEUP: return FromScanCode(kVK_PageUp);
        case WXK_PAGEDOWN: return FromScanCode(kVK_PageDown);
        default: return (key>=32 && key<127) ? FromChar(static_cast<char>(key)) : KeyboardCode();
    }
}
#if defined(DEBUG)
void KeyboardCode::Dump() const {}
#endif
