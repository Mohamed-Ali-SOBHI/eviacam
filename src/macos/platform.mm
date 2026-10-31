// SPDX-License-Identifier: GPL-3.0-or-later
#include "platform.h"
#import <Cocoa/Cocoa.h>
#import <AVFoundation/AVFoundation.h>
#import <ApplicationServices/ApplicationServices.h>
#include <atomic>
#include <memory>
namespace eviacam {
bool MacCameraPermission() {
    @autoreleasepool {
        auto status = [AVCaptureDevice authorizationStatusForMediaType:AVMediaTypeVideo];
        if (status == AVAuthorizationStatusNotDetermined) {
            // Keep the main run loop alive while the native consent dialog is shown.
            auto finished = std::make_shared<std::atomic<bool>>(false);
            [AVCaptureDevice requestAccessForMediaType:AVMediaTypeVideo completionHandler:^(BOOL) {
                finished->store(true);
            }];
            while (!finished->load()) CFRunLoopRunInMode(kCFRunLoopDefaultMode, 0.05, true);
            status = [AVCaptureDevice authorizationStatusForMediaType:AVMediaTypeVideo];
        }
        return status == AVAuthorizationStatusAuthorized;
    }
}
bool MacAccessibilityPermission(bool prompt) {
    NSDictionary* options = @{(__bridge NSString*)kAXTrustedCheckOptionPrompt: @(prompt)};
    return AXIsProcessTrustedWithOptions((__bridge CFDictionaryRef)options);
}
std::vector<std::string> MacCameraNames() {
    @autoreleasepool {
        std::vector<std::string> names;
        // Same order/API as OpenCV 4.10's AVFoundation backend; enumeration
        // never creates a capture session or opens the camera.
        NSArray* devices = [[AVCaptureDevice devicesWithMediaType:AVMediaTypeVideo]
            arrayByAddingObjectsFromArray:[AVCaptureDevice devicesWithMediaType:AVMediaTypeMuxed]];
        devices = [devices sortedArrayUsingComparator:^NSComparisonResult(AVCaptureDevice* a, AVCaptureDevice* b) {
            return [a.uniqueID compare:b.uniqueID];
        }];
        for (AVCaptureDevice* device in devices)
            names.emplace_back([[device localizedName] UTF8String]);
        return names;
    }
}
void MacConfigureClickWindow(void* handle, bool acceptsFocus) {
    NSWindow* window = [(NSView*)handle window];
    [window setLevel:NSFloatingWindowLevel];
    [window setCollectionBehavior:NSWindowCollectionBehaviorCanJoinAllSpaces];
    if ([window isKindOfClass:[NSPanel class]]) {
        NSPanel* panel = (NSPanel*)window;
        [panel setBecomesKeyOnlyIfNeeded:!acceptsFocus];
        if (!acceptsFocus) [panel setStyleMask:([panel styleMask] | NSWindowStyleMaskNonactivatingPanel)];
    }
}
}
