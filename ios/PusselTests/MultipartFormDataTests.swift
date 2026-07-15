import XCTest

@testable import Pussel

final class MultipartFormDataTests: XCTestCase {
  func testFilePartStructure() {
    var form = MultipartFormData(boundary: "TESTBOUNDARY")
    form.appendFile(
      name: "file", filename: "puzzle.jpg", mimeType: "image/jpeg", data: Data("JPEGBYTES".utf8))
    let body = String(decoding: form.encoded(), as: UTF8.self)

    let expected =
      "--TESTBOUNDARY\r\n"
      + "Content-Disposition: form-data; name=\"file\"; filename=\"puzzle.jpg\"\r\n"
      + "Content-Type: image/jpeg\r\n\r\n"
      + "JPEGBYTES\r\n"
      + "--TESTBOUNDARY--\r\n"
    XCTAssertEqual(body, expected)
    XCTAssertEqual(form.contentType, "multipart/form-data; boundary=TESTBOUNDARY")
  }

  func testFieldPartStructure() {
    var form = MultipartFormData(boundary: "TESTBOUNDARY")
    form.appendField(name: "corners", value: "{\"x\":0.1}")
    let body = String(decoding: form.encoded(), as: UTF8.self)

    XCTAssertTrue(
      body.contains("Content-Disposition: form-data; name=\"corners\"\r\n\r\n{\"x\":0.1}\r\n"))
    XCTAssertTrue(body.hasSuffix("--TESTBOUNDARY--\r\n"))
  }

  func testBinaryDataPreserved() {
    let binary = Data([0xFF, 0xD8, 0x00, 0x1F, 0xFF])
    var form = MultipartFormData(boundary: "B")
    form.appendFile(name: "file", filename: "f.jpg", mimeType: "image/jpeg", data: binary)
    XCTAssertNotNil(form.encoded().range(of: binary))
  }
}
