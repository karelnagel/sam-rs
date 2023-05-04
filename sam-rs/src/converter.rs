#[cfg(test)]
mod test {
    use burn::{
        module::Module,
        record::{Record, SentitiveCompactRecordSettings},
    };

    use crate::{
        build_sam::{build_sam_test, build_sam_vit_h},
        tests::helpers::{load_module, TestBackend},
    };

    #[ignore]
    #[test]
    fn convert_sam_test() {
        let mut sam = build_sam_test::<TestBackend>(None);
        sam = load_module("sam_test", sam);
        sam.into_record()
            .record::<SentitiveCompactRecordSettings>("sam_test".into())
            .unwrap();
    }

    #[ignore]
    #[test]
    fn convert_sam_vit_h() {
        let mut sam = build_sam_vit_h::<TestBackend>(None);
        sam = load_module("sam_vit_h", sam);
        sam.into_record()
            .record::<SentitiveCompactRecordSettings>("sam_vit_h.bin".into())
            .unwrap();
    }
}
