// SPDX-License-Identifier: GPL-3.0-or-later
#include "mouseevents.h"
#include <cmath>
namespace eviacam {
CGEventRef MacMouseEvents::Create(long x, long y, int flags, CGPoint point, double now) {
    CGEventType type = kCGEventMouseMoved;
    CGMouseButton button = kCGMouseButtonLeft;
    // eViacam's existing cross-platform mouse command values.
    switch (flags) {
        case 0x8000: point = CGPointMake(x,y); break;
        case 0: point = CGPointMake(point.x+x,point.y+y); break;
        case 2:
            type = kCGEventLeftMouseDown; leftDown = true;
            clickCount = (lastLeftClick>=0 && now-lastLeftClick<0.5 &&
                std::abs(point.x-lastClickPoint.x)<4 && std::abs(point.y-lastClickPoint.y)<4) ? clickCount+1 : 1;
            lastLeftClick=now; lastClickPoint=point; break;
        case 4: type=kCGEventLeftMouseUp; leftDown=false; break;
        case 8: type=kCGEventRightMouseDown; button=kCGMouseButtonRight; rightDown=true; break;
        case 16: type=kCGEventRightMouseUp; button=kCGMouseButtonRight; rightDown=false; break;
        case 32: type=kCGEventOtherMouseDown; button=kCGMouseButtonCenter; middleDown=true; break;
        case 64: type=kCGEventOtherMouseUp; button=kCGMouseButtonCenter; middleDown=false; break;
        default: return nullptr;
    }
    if (type==kCGEventMouseMoved) {
        if (leftDown) type=kCGEventLeftMouseDragged;
        else if (rightDown) { type=kCGEventRightMouseDragged; button=kCGMouseButtonRight; }
        else if (middleDown) { type=kCGEventOtherMouseDragged; button=kCGMouseButtonCenter; }
    }
    CGEventRef event=CGEventCreateMouseEvent(nullptr,type,point,button);
    if (event && (type==kCGEventLeftMouseDown || type==kCGEventLeftMouseUp))
        CGEventSetIntegerValueField(event,kCGMouseEventClickState,clickCount);
    if (event && flags==0) {
        CGEventSetIntegerValueField(event,kCGMouseEventDeltaX,x);
        CGEventSetIntegerValueField(event,kCGMouseEventDeltaY,y);
    }
    return event;
}
}
