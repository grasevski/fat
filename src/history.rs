//! Historical data loader.
use super::fxcm;
use arrayvec::ArrayString;
use chrono::{Datelike, Duration, NaiveDate};
use csv::{DeserializeRecordsIntoIter, Reader};
use enum_map::EnumMap;
use flate2::read::GzDecoder;
use std::{fmt::Write, io::Read};

/// Iterator to get historical data for all symbols.
pub struct History<R: Read, F: FnMut(&str) -> fxcm::Result<R>> {
    /// Starting point of time range.
    begin: NaiveDate,

    /// Optional end point of time range.
    end: Option<NaiveDate>,

    /// Candle interval.
    frequency: fxcm::Frequency,

    /// Interface to fetch historical data.
    client: F,

    /// A state machine for each symbol.
    buf: EnumMap<fxcm::Symbol, (Option<fxcm::Candle>, Option<HistoryLoader<R>>)>,
}

impl<R: Read, F: FnMut(&str) -> fxcm::Result<R>> History<R, F> {
    /// Initializes data loader for given date range.
    pub fn new(
        mut client: F,
        begin: NaiveDate,
        end: Option<NaiveDate>,
        frequency: fxcm::Frequency,
    ) -> fxcm::Result<Self> {
        let mut buf = EnumMap::default();
        for (k, (candle, loader)) in &mut buf {
            let mut l = HistoryLoader::new(begin);
            if let Some(c) = l.next(k, begin, end, frequency, &mut client) {
                *candle = Some(c?);
                *loader = Some(l);
            } else {
                *loader = None;
            }
        }
        Ok(Self {
            begin,
            end,
            frequency,
            client,
            buf,
        })
    }
}

impl<R: Read, F: FnMut(&str) -> fxcm::Result<R>> Iterator for History<R, F> {
    type Item = fxcm::FallibleCandle;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(candle) = self.buf.values().flat_map(|x| x.0.clone()).min() {
            let (ref mut c, ref mut l) = self.buf[candle.symbol];
            if let Some(loader) = l {
                if let Some(x) = loader.next(
                    candle.symbol,
                    self.begin,
                    self.end,
                    self.frequency,
                    &mut self.client,
                ) {
                    match x {
                        Ok(x) => {
                            *c = Some(x);
                        }
                        Err(err) => {
                            return Some(Err(err));
                        }
                    }
                } else {
                    *c = None;
                    *l = None;
                }
            } else {
                *c = None;
            }
            Some(Ok(candle))
        } else {
            None
        }
    }
}

/// A data loader for a given symbol.
struct HistoryLoader<R: Read> {
    /// Current time point in the data.
    current: NaiveDate,

    /// Whether the current response is empty.
    empty: bool,

    /// Handle for the current candle data response file.
    rdr: Option<DeserializeRecordsIntoIter<GzDecoder<R>, fxcm::Historical>>,
}

impl<R: Read> HistoryLoader<R> {
    /// Initializes the data loader with the start date.
    fn new(current: NaiveDate) -> Self {
        Self {
            current,
            empty: true,
            rdr: None,
        }
    }

    /// Gets the next candle record from the data source.
    fn next(
        &mut self,
        symbol: fxcm::Symbol,
        begin: NaiveDate,
        end: Option<NaiveDate>,
        frequency: fxcm::Frequency,
        mut client: impl FnMut(&str) -> fxcm::Result<R>,
    ) -> Option<fxcm::FallibleCandle> {
        if self.rdr.is_none() {
            const ENDPOINT: &str = "https://candledata.fxcorporate.com";
            let mut url = ArrayString::<64>::new();
            if let Err(err) = if let fxcm::Frequency::Daily = frequency {
                self.current -= Duration::days(self.current.ordinal0().into());
                if let Some(end) = end {
                    if self.current >= end {
                        return None;
                    }
                }
                let year = self.current.year();
                self.current += Duration::weeks(53);
                write!(&mut url, "{}/D1/{}/{}.csv.gz", ENDPOINT, symbol, year)
            } else {
                self.current -= Duration::days((self.current.ordinal0() % 7).into());
                if let Some(end) = end {
                    if self.current >= end {
                        return None;
                    }
                }
                let year = self.current.year();
                let week =
                    1 + self.current.ordinal0() / 7 + u32::from(self.current.ordinal0() % 7 != 0);
                let frequency = if let fxcm::Frequency::Minutely = frequency {
                    "m1"
                } else {
                    "H1"
                };
                self.current += Duration::weeks(1);
                write!(
                    &mut url,
                    "{}/{}/{}/{}/{}.csv.gz",
                    ENDPOINT, frequency, symbol, year, week
                )
            } {
                return Some(Err(fxcm::Error::from(err)));
            }
            match client(url.as_str()) {
                Ok(res) => {
                    let rdr = Reader::from_reader(GzDecoder::new(res));
                    self.rdr = Some(rdr.into_deserialize());
                }
                Err(res) => {
                    return Some(Err(res));
                }
            }
        }
        if let Some(rdr) = &mut self.rdr {
            for candle in rdr {
                self.empty = false;
                match candle {
                    Ok(candle) => match fxcm::Candle::new(symbol, candle) {
                        Ok(candle) => {
                            let t = candle.ts.naive_utc().date();
                            if t >= begin {
                                return if let Some(end) = end {
                                    if t < end {
                                        Some(Ok(candle))
                                    } else {
                                        None
                                    }
                                } else {
                                    Some(Ok(candle))
                                };
                            }
                        }
                        Err(err) => {
                            return Some(Err(err));
                        }
                    },
                    Err(ret) => {
                        return Some(Err(fxcm::Error::from(ret)));
                    }
                }
            }
            if self.empty {
                None
            } else {
                self.rdr = None;
                self.empty = true;
                self.next(symbol, begin, end, frequency, client)
            }
        } else {
            None
        }
    }
}
